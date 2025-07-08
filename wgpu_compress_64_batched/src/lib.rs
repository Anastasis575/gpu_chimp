mod compute_s_shader;
mod cpu;
mod decompressor;
mod final_compress;
mod finalize;

use crate::compute_s_shader::{ComputeS, ComputeSImpl64};
use crate::cpu::finalize::CPUImpl;
use crate::final_compress::{FinalCompress, FinalCompressImpl64};
use crate::finalize::{Finalize, Finalizer};
use anyhow::Result;
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor64};
use compress_utils::general_utils::{
    add_padding_to_fit_buffer_count_f64, get_buffer_size, Padding,
};
use compress_utils::time_it;
use compress_utils::types::{ChimpOutput, S};
use log::info;
use pollster::FutureExt;
use std::sync::Arc;

#[derive(Debug)]
pub enum FinalizerEnum {
    GPU,
    CPU,
}
#[derive(Debug)]
pub enum FinalizerImpl<'a> {
    GPU(Finalizer<'a>),
    CPU(CPUImpl),
}

#[async_trait]
impl Finalize for FinalizerImpl<'_> {
    async fn finalize(
        &self,
        chimp_output: &mut Vec<ChimpOutput>,
        padding: usize,
    ) -> Result<Vec<u8>> {
        match self {
            FinalizerImpl::GPU(impll) => impll.finalize(chimp_output, padding).await,
            FinalizerImpl::CPU(impll) => impll.finalize(chimp_output, padding).await,
        }
    }
}
#[derive(Debug)]
pub struct ChimpCompressorBatched {
    debug: bool,
    context: Arc<Context>,
    finalizer: FinalizerEnum,
}

impl Default for ChimpCompressorBatched {
    fn default() -> Self {
        Self {
            debug: false,
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            finalizer: FinalizerEnum::GPU,
        }
    }
}

#[async_trait]
impl Compressor64 for ChimpCompressorBatched {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError> {
        let mut padding = Padding(0);
        let buffer_size = get_buffer_size();
        let mut values = vec.to_owned();
        values = add_padding_to_fit_buffer_count_f64(values, buffer_size, &mut padding);
        let mut total_millis = 0;
        let mut s_values: Vec<S>;
        let mut chimp_vec: Vec<ChimpOutput>;

        let compute_s_impl = self.compute_s_factory();
        let final_compress_impl = self.compute_final_compress_factory();
        let finalize_impl = self.compute_finalize_factory();

        let output_vec: BitVec;
        time_it!(
            {
                s_values = compute_s_impl.compute_s(&mut values).await?;
            },
            total_millis,
            "s computation stage"
        );
        time_it!(
            {
                chimp_vec = final_compress_impl
                    .final_compress(&mut values, &mut s_values, 0)
                    .await?;
            },
            total_millis,
            "final output stage"
        );
        time_it!(
            {
                output_vec = BitVec::from_bytes(
                    finalize_impl
                        .finalize(&mut chimp_vec, padding.0)
                        .await?
                        .as_slice(),
                );
            },
            total_millis,
            "final Result collection"
        );

        Ok(output_vec.to_bytes())
    }
}

impl ChimpCompressorBatched {
    pub fn new(debug: bool, context: Arc<Context>, finalizer: FinalizerEnum) -> Self {
        Self {
            debug,
            context,
            finalizer,
        }
    }
    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }
    pub fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }
    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn debug(&self) -> bool {
        self.debug
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    fn compute_s_factory(&self) -> impl ComputeS + use<'_> {
        ComputeSImpl64::new(self.context())
    }
    fn compute_final_compress_factory(&self) -> impl FinalCompress + use<'_> {
        FinalCompressImpl64::new(self.context(), self.debug())
    }
    fn compute_finalize_factory(&self) -> impl Finalize + use<'_> {
        match self.finalizer {
            FinalizerEnum::GPU => FinalizerImpl::GPU(Finalizer::new(self.context())),
            FinalizerEnum::CPU => FinalizerImpl::CPU(CPUImpl::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cpu::decompressor;
    use crate::ChimpCompressorBatched;
    use crate::FinalizerEnum::{CPU, GPU};
    use compress_utils::context::Context;
    use compress_utils::cpu_compress::{Compressor64, Decompressor64};
    use compress_utils::general_utils::check_for_debug_mode;
    use decompressor::BatchedDecompressorCpu64;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::sync::Arc;
    use std::{env, fs};
    use tracing_subscriber::util::SubscriberInitExt;

    fn get_third(field: &str) -> Option<String> {
        field
            .split(",")
            .collect_vec()
            .get(2)
            .map(|it| it.to_string())
    }
    //noinspection ALL
    fn get_values() -> anyhow::Result<Vec<f64>> {
        let dir = env::current_dir()?;
        let file_path = dir.parent().unwrap().join("city_temperature.csv");
        let file_txt = fs::read_to_string(file_path)?;
        let values = file_txt
            .split("\n")
            .map(get_third)
            .filter(|p| p.is_some())
            .map(|s| s.unwrap().parse::<f64>().unwrap())
            .collect_vec()
            .to_vec();
        Ok(values)
    }
    //noinspection DuplicatedCode
    #[test]
    fn test_matching_outputs() {
        // let value_count = 0..(256 * 3);

        let subscriber = tracing_subscriber::fmt()
            .compact()
            .with_env_filter("wgpu_compress_64=info")
            // .with_writer(
            //     OpenOptions::new()
            //         .create(true)
            //         .truncate(true)
            //         .write(true)
            //         .open("run.log")
            //         .unwrap(),
            // )
            .finish();
        subscriber.init();

        let mut values = get_values().expect("Could not read test values").to_vec();
        log::info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched::default();
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let compressed_values1 = compressor.compress(&mut values).block_on().unwrap();

        let mut values = get_values().expect("Could not read test values").to_vec();
        log::info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched {
            finalizer: CPU,
            ..ChimpCompressorBatched::default()
        };
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let compressed_values2 = compressor.compress(&mut values).block_on().unwrap();
        assert_eq!(compressed_values2, compressed_values1);
    }
    //noinspection DuplicatedCode
    #[test]
    fn test_decompress_able_64() {
        // let value_count = 0..(256 * 109);
        let subscriber = tracing_subscriber::fmt()
            .compact()
            .with_env_filter("wgpu_compress_32=info")
            // .with_writer(
            //     OpenOptions::new()
            //         .create(true)
            //         .truncate(true)
            //         .write(true)
            //         .open("run.log")
            //         .unwrap(),
            // )
            .finish();
        subscriber.init();

        let mut values = get_values().expect("Could not read test values")[0..512].to_vec();
        log::info!("Starting compression of {} values", values.len());
        let context = Arc::new(
            Context::initialize_with_adapter("NVIDIA".to_string())
                .block_on()
                .unwrap(),
        );
        let mut compressor = ChimpCompressorBatched {
            finalizer: GPU,
            context: context.clone(),
            ..ChimpCompressorBatched::default()
        };
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let mut compressed_values2 = compressor.compress(&mut values).block_on().unwrap();
        // let compressed_values3 = compressor.compress(&mut values).block_on().unwrap();

        // assert_eq!(compressed_values2, compressed_values3);

        let decompressor = BatchedDecompressorCpu64::default();
        match decompressor.decompress(&mut compressed_values2).block_on() {
            Ok(decompressed_values) => {
                // fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                // fs::write("expected.log", values.iter().join("\n")).unwrap();
                assert_eq!(decompressed_values, values)
            }
            Err(err) => {
                eprintln!("Decompression error: {:?}", err);
                panic!("{}", err);
            }
        }
        // let value_count = 0..(256 * 109);
    }
}
