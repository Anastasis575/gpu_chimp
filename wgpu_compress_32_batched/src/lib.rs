mod compute_s_shader;
pub mod cpu;
pub mod decompressor;
mod final_compress;
mod finalize;

use crate::compute_s_shader::{ComputeS, ComputeSImpl};
use crate::cpu::finalize::CPUImpl;
use crate::final_compress::{FinalCompress, FinalCompressImpl};
use crate::finalize::{Finalize, Finalizer};
use anyhow::Result;
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::{add_padding_to_fit_buffer_count, ChimpBufferInfo, Padding};
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
pub enum FinalizerImpl {
    GPU(Finalizer),
    CPU(CPUImpl),
}

#[async_trait]
impl Finalize for FinalizerImpl {
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
impl Compressor<f32> for ChimpCompressorBatched {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>, CompressionError> {
        let mut padding = Padding(0);
        let buffer_size = ChimpBufferInfo::get().buffer_size();
        let mut values = vec.to_owned();
        values = add_padding_to_fit_buffer_count(values, buffer_size, &mut padding);
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
    /// Processes the number of iterations required to handle the maximum buffer size for the given `vec`.
    ///
    /// # Parameters
    /// - `vec`: A mutable reference to a `Vec<f32>`. This vector is used for calculations to determine
    ///   how many iterations are needed to stay within the maximum buffer size.
    ///
    /// # Returns
    /// - Returns the number of iterations required to stay within the maximum allowed storage buffer size.
    ///
    /// # Constants
    /// - `BUFFER_MAX_SIZE`: A constant defining the maximum storage buffer size as `134217728` bytes
    ///   (128 MiB). This is the maximum amount of data one storage buffer can hold.
    ///
    /// # Calculations
    /// - The function calculates the number of iterations by dividing the total size of the data in
    ///   the vector (`vec.len() * size_of::<S>()`) by the `BUFFER_MAX_SIZE`.
    /// - If the result is less than `1`, it defaults to returning `1` to guarantee at least one iteration.
    fn process_iterations_for_max_buffer_size(&self, vec: &mut Vec<f32>) -> usize {
        const BUFFER_MAX_SIZE: usize = 134217728;
        (vec.len() * size_of::<S>()) / BUFFER_MAX_SIZE + 1 //the S buffers are the most costly to allocate
    }
    fn process_iterations_for_max_workgroup_count(&self, vec: &mut Vec<f32>) -> usize {
        let max_workgroup_count = self.context.get_max_workgroup_size();
        vec.len() / max_workgroup_count + 1
    }
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
    pub fn context_a(&self) -> Arc<Context> {
        self.context.clone()
    }

    pub fn debug(&self) -> bool {
        self.debug
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    fn compute_s_factory(&self) -> impl ComputeS {
        ComputeSImpl::new(self.context_a())
    }
    fn compute_final_compress_factory(&self) -> impl FinalCompress + use<'_> {
        FinalCompressImpl::new(self.context_a(), self.debug())
    }
    fn compute_finalize_factory(&self) -> impl Finalize + use<'_> {
        match self.finalizer {
            FinalizerEnum::GPU => FinalizerImpl::GPU(Finalizer::new(self.context_a())),
            FinalizerEnum::CPU => FinalizerImpl::CPU(CPUImpl::default()),
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::decompressor::BatchedGPUDecompressor;
    use crate::FinalizerEnum::{CPU, GPU};
    use crate::{cpu, ChimpCompressorBatched};
    use compress_utils::context::Context;
    use compress_utils::cpu_compress::{Compressor, Decompressor};
    use compress_utils::general_utils::check_for_debug_mode;
    use cpu::decompressor;
    use indicatif::ProgressIterator;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;
    use std::{env, fs, os};
    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::util::SubscriberInitExt;

    fn get_third(field: &str) -> Option<String> {
        field
            .split(",")
            .collect_vec()
            .get(2)
            .map(|it| it.to_string())
    }
    //noinspection ALL
    fn get_values(file_name: impl Into<String>) -> anyhow::Result<Vec<f32>> {
        let dir = env::current_dir()?;
        let file_path = dir.parent().unwrap().join(file_name.into());
        let file_txt = fs::read_to_string(file_path)?;
        let values = file_txt
            .split("\n")
            .map(get_third)
            .filter(|p| p.is_some())
            .map(|s| s.unwrap().parse::<f32>().unwrap())
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

        let mut values = get_values("city_temperature.csv")
            .expect("Could not read test values")
            .to_vec();
        log::info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched::default();
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let compressed_values1 = compressor.compress(&mut values).block_on().unwrap();

        let mut values = get_values("city_temperature.csv")
            .expect("Could not read test values")
            .to_vec();
        log::info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched {
            finalizer: GPU,
            ..ChimpCompressorBatched::default()
        };
        // if check_for_debug_mode().expect("Could not read file system") {
        //     compressor.set_debug(true);
        // }
        let compressed_values2 = compressor.compress(&mut values).block_on().unwrap();
        assert_eq!(compressed_values2, compressed_values1);
    }
    //noinspection DuplicatedCode
    #[test]
    fn test_decompress_able() {
        let context = Arc::new(
            Context::initialize_with_adapter("NVIDIA".to_string())
                .block_on()
                .unwrap(),
        );
        env::set_var("CHIMP_BUFFER_SIZE", "1024".to_string());

        for file_name in vec![
            "city_temperature.csv",
            "SSD_HDD_benchmarks.csv",
            "Stocks-Germany-sample.txt",
        ]
        .into_iter()
        {
            println!("{file_name}");
            let filename = format!("{}_chimp64_output.txt", &file_name);
            if fs::exists(&filename).unwrap() {
                fs::remove_file(&filename).unwrap();
            }
            let mut messages = Vec::<String>::with_capacity(30);
            let mut values = get_values(file_name)
                .expect("Could not read test values")
                .to_vec();
            for size_checkpoint in (1..11).progress() {
                let mut value_new = values[0..(values.len() * size_checkpoint) / 10].to_vec();
                log::info!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let mut compressor = ChimpCompressorBatched::new(false, context.clone(), GPU);
                let mut compressed_values2 =
                    compressor.compress(&mut value_new).block_on().unwrap();
                let compression_time = time.elapsed().as_millis();

                const SIZE_IN_BYTE: usize = 8;
                let compression_ratio =
                    (compressed_values2.len() * SIZE_IN_BYTE) as f64 / value_new.len() as f64;
                messages.push(format!(
                    "Compression ratio {size_checkpoint}0% {compression_ratio}\n"
                ));
                messages.push(format!(
                    "Encoding time {size_checkpoint}0%: {compression_time}\n"
                ));

                let time = std::time::Instant::now();
                let decompressor = BatchedGPUDecompressor::new(context.clone());
                match decompressor.decompress(&mut compressed_values2).block_on() {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();
                        messages.push(format!(
                            "Decoding time {size_checkpoint}0%: {decompression_time}\n"
                        ));
                        fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        fs::write("expected.log", values.iter().join("\n")).unwrap();
                        assert_eq!(decompressed_values, value_new);
                    }
                    Err(err) => {
                        eprintln!("Decompression error: {:?}", err);
                        panic!("{}", err);
                    }
                }
            }
            let f = OpenOptions::new()
                .append(true)
                .create(true)
                .open(file_name)
                .expect("temp");
            let mut fw = f.make_writer();
            for message in messages {
                write!(fw, "{message}").unwrap()
            }
        }
    }
    #[test]
    fn test_decompress_able_buffer() {
        let context = Arc::new(
            Context::initialize_with_adapter("NVIDIA".to_string())
                .block_on()
                .unwrap(),
        );
        for buffer_size in vec![256, 512, 1024, 2048].into_iter() {
            let filename = format!("{}_chimp64_output.txt", buffer_size);
            if fs::exists(&filename).unwrap() {
                fs::remove_file(&filename).unwrap();
            }
            env::set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
            println!("Buffer size: {}", env::var("CHIMP_BUFFER_SIZE").unwrap());
            let mut messages = Vec::<String>::with_capacity(30);
            let mut values = get_values("city_temperature.csv")
                .expect("Could not read test values")
                .to_vec();
            for size_checkpoint in (1..11).progress() {
                let mut value_new = values[0..(values.len() * size_checkpoint) / 10].to_vec();
                log::info!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let mut compressor = ChimpCompressorBatched::new(false, context.clone(), GPU);
                let mut compressed_values2 =
                    compressor.compress(&mut value_new).block_on().unwrap();
                let compression_time = time.elapsed().as_millis();

                const SIZE_IN_BYTE: usize = 8;
                let compression_ratio =
                    (compressed_values2.len() * SIZE_IN_BYTE) as f64 / value_new.len() as f64;
                messages.push(format!(
                    "Compression ratio {size_checkpoint}0% {compression_ratio}\n"
                ));
                messages.push(format!(
                    "Encoding time {size_checkpoint}0%: {compression_time}\n"
                ));

                let time = std::time::Instant::now();
                let decompressor = BatchedGPUDecompressor::new(context.clone());
                match decompressor.decompress(&mut compressed_values2).block_on() {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();
                        messages.push(format!(
                            "Decoding time {size_checkpoint}0%: {decompression_time}\n"
                        ));
                        fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        fs::write("expected.log", values.iter().join("\n")).unwrap();
                        assert_eq!(decompressed_values, value_new);
                    }
                    Err(err) => {
                        eprintln!("Decompression error: {:?}", err);
                        panic!("{}", err);
                    }
                }
            }
            let f = OpenOptions::new()
                .append(true)
                .create(true)
                .open(filename)
                .expect("temp");
            let mut fw = f.make_writer();
            for message in messages {
                write!(fw, "{message}").unwrap()
            }
        }
    }
    #[test]
    fn test_decompress_able_old() {
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

        let mut values = get_values("city_temperature.csv")
            .expect("Could not read test values")
            .to_vec();
        values.extend(
            get_values("city_temperature.csv")
                .expect("Could not read test values")
                .to_vec(),
        );
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

        let decompressor = BatchedGPUDecompressor::new(context);
        match decompressor.decompress(&mut compressed_values2).block_on() {
            Ok(decompressed_values) => {
                fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                fs::write("expected.log", values.iter().join("\n")).unwrap();
                assert_eq!(decompressed_values, values)
            }
            Err(err) => {
                eprintln!("Decompression error: {:?}", err);
                panic!("{}", err);
            }
        }
    }
}
