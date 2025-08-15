use crate::calculate_indexes::{CalculateIndexes64, GPUCalculateIndexes64};
use crate::compute_s_shader::{ComputeS, ComputeSImpl};
use crate::final_compress::{FinalCompress, FinalCompressImpl64};
use crate::finalize::{Finalize, Finalizer64};
use anyhow::Result;
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::{ChimpBufferInfo, DeviceEnum, MaxGroupGnostic, Padding};
use compress_utils::time_it;
use compress_utils::types::{ChimpOutput64, S};
use log::info;
use pollster::FutureExt;
use std::ops::Div;
use std::sync::Arc;

mod calculate_indexes;
mod compute_s_shader;
pub mod cpu;
pub mod decompressor;
mod final_compress;
mod finalize;

#[derive(Debug)]
pub struct ChimpCompressorBatched64 {
    context: Arc<Context>,
    device_type: DeviceEnum,
}
impl Default for ChimpCompressorBatched64 {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            device_type: DeviceEnum::GPU,
        }
    }
}

pub fn add_padding_to_fit_buffer_count_64(
    mut values: Vec<f64>,
    buffer_size: usize,
    padding: &mut Padding,
) -> Vec<f64> {
    if values.len() % buffer_size != 0 {
        let count = (values.len().div(buffer_size) + 1) * buffer_size - values.len();
        padding.0 = count;
        for _i in 0..count {
            values.push(0f64);
        }
    }
    values
}

#[async_trait]
impl Compressor<f64> for ChimpCompressorBatched64 {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError> {
        let mut padding = Padding(0);
        let buffer_size = ChimpBufferInfo::get().buffer_size();
        let mut values = vec.to_owned();
        values = add_padding_to_fit_buffer_count_64(values, buffer_size, &mut padding);
        let mut total_millis = 0;
        let mut s_values: Vec<S>;
        let mut chimp_vec: Vec<ChimpOutput64>;
        let mut indexes: Vec<u32>;

        let compute_s_impl = self.compute_s_factory();
        let final_compress_impl = self.compute_final_compress_factory();
        let indexes_impl = self.calculate_index_factory();
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
            "compression stage"
        );
        time_it!(
            {
                indexes = indexes_impl
                    .calculate_indexes(&mut chimp_vec, ChimpBufferInfo::get().buffer_size() as u32)
                    .await?;
            },
            total_millis,
            "calculate trim size stage"
        );
        time_it!(
            {
                output_vec = BitVec::from_bytes(
                    finalize_impl
                        .finalize(&mut chimp_vec, padding.0, indexes)
                        .await?
                        .as_slice(),
                );
            },
            total_millis,
            "trimming stage"
        );
        Ok(output_vec.to_bytes())
    }
}
enum ComputeS64Impls {
    GPU(ComputeSImpl),
    CPU(cpu::compute_s::CpuComputeSImpl),
}

impl MaxGroupGnostic for ComputeS64Impls {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        match self {
            ComputeS64Impls::GPU(c) => c.get_max_number_of_groups(content_len),
            ComputeS64Impls::CPU(c) => c.get_max_number_of_groups(content_len),
        }
    }
}

#[async_trait]
impl ComputeS for ComputeS64Impls {
    async fn compute_s(&self, values: &mut [f64]) -> anyhow::Result<Vec<S>> {
        match self {
            ComputeS64Impls::GPU(c) => c.compute_s(values).await,
            ComputeS64Impls::CPU(c) => c.compute_s(values).await,
        }
    }
}

enum CalculateIndexesImpls {
    GPU(GPUCalculateIndexes64),
    CPU(cpu::calculate_indexes::CPUCalculateIndexes64),
}
#[async_trait]
impl CalculateIndexes64 for CalculateIndexesImpls {
    async fn calculate_indexes(&self, input: &[ChimpOutput64], size: u32) -> Result<Vec<u32>> {
        match self {
            CalculateIndexesImpls::GPU(c) => c.calculate_indexes(input, size).await,
            CalculateIndexesImpls::CPU(c) => c.calculate_indexes(input, size).await,
        }
    }
}

enum Compress64Impls {
    GPU(FinalCompressImpl64),
    CPU(cpu::chimp_compress::CPUFinalCompressImpl64),
}

impl MaxGroupGnostic for Compress64Impls {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        match self {
            Compress64Impls::GPU(c) => c.get_max_number_of_groups(content_len),
            Compress64Impls::CPU(c) => c.get_max_number_of_groups(content_len),
        }
    }
}

#[async_trait]
impl FinalCompress for Compress64Impls {
    async fn final_compress(
        &self,
        input: &mut Vec<f64>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> anyhow::Result<Vec<ChimpOutput64>> {
        match self {
            Compress64Impls::GPU(c) => c.final_compress(input, s_values, padding).await,
            Compress64Impls::CPU(c) => c.final_compress(input, s_values, padding).await,
        }
    }
}

enum Finalizer64impls {
    GPU(Finalizer64),
    CPU(cpu::finalize::CPUFinalizer64),
}
#[async_trait]
impl Finalize for Finalizer64impls {
    async fn finalize(
        &self,
        chimp_output: &mut Vec<ChimpOutput64>,
        padding: usize,
        indexes: Vec<u32>,
    ) -> anyhow::Result<Vec<u8>> {
        match self {
            Finalizer64impls::GPU(f) => f.finalize(chimp_output, padding, indexes).await,
            Finalizer64impls::CPU(f) => f.finalize(chimp_output, padding, indexes).await,
        }
    }
}
#[allow(unused)]
impl ChimpCompressorBatched64 {
    fn compute_s_factory(&self) -> ComputeS64Impls {
        match self.device_type() {
            DeviceEnum::GPU => ComputeS64Impls::GPU(ComputeSImpl::new(self.context.clone())),
            DeviceEnum::CPU => {
                ComputeS64Impls::CPU(cpu::compute_s::CpuComputeSImpl::new(self.context.clone()))
            }
        }
    }
    fn compute_final_compress_factory(&self) -> Compress64Impls {
        match self.device_type() {
            &DeviceEnum::GPU => {
                Compress64Impls::GPU(FinalCompressImpl64::new(self.context.clone(), false))
            }
            &DeviceEnum::CPU => Compress64Impls::CPU(
                cpu::chimp_compress::CPUFinalCompressImpl64::new(self.context.clone(), false),
            ),
        }
    }
    fn compute_finalize_factory(&self) -> Finalizer64impls {
        match self.device_type() {
            &DeviceEnum::GPU => Finalizer64impls::GPU(Finalizer64::new(self.context.clone())),
            &DeviceEnum::CPU => {
                Finalizer64impls::CPU(cpu::finalize::CPUFinalizer64::new(self.context.clone()))
            }
        }
    }
    fn calculate_index_factory(&self) -> CalculateIndexesImpls {
        match self.device_type() {
            &DeviceEnum::GPU => {
                CalculateIndexesImpls::GPU(GPUCalculateIndexes64::new(self.context.clone()))
            }
            &DeviceEnum::CPU => CalculateIndexesImpls::CPU(
                cpu::calculate_indexes::CPUCalculateIndexes64::new(self.context.clone()),
            ),
        }
    }
    pub(crate) fn new(context: impl Into<Arc<Context>>) -> Self {
        Self {
            context: context.into(),
            device_type: DeviceEnum::GPU,
        }
    }

    pub(crate) fn with_device(self, device: impl Into<DeviceEnum>) -> Self {
        Self {
            device_type: device.into(),
            ..self
        }
    }
    pub(crate) fn device_type(&self) -> &DeviceEnum {
        &self.device_type
    }
}

fn splitter(value: f64) -> [f32; 2] {
    let bits = bytemuck::cast::<f64, u64>(value);
    [
        f32::from_bits((bits >> 32) as u32),
        f32::from_bits(bits as u32),
    ]
}
fn merger(value: [f32; 2]) -> f64 {
    let high = (value[0].to_bits() as u64) << 32;
    let low = value[1].to_bits() as u64;
    bytemuck::cast(high | low)
}

#[cfg(test)]
mod tests {
    use crate::{decompressor, merger, splitter, ChimpCompressorBatched64};
    use compress_utils::context::Context;
    use compress_utils::cpu_compress::{Compressor, Decompressor};
    use compress_utils::general_utils::DeviceEnum::CPU;
    use env::set_var;
    use indicatif::ProgressIterator;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;
    use std::{env, fs};
    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::util::SubscriberInitExt;

    #[test]
    fn splitter_merger() {
        let original = 123.456789_f64;
        let split = splitter(original);
        let merged = merger(split);
        assert_eq!(original, merged);
    }

    #[test]
    fn splitter_merger2() {
        let original = 25.0003;
        let split = splitter(original);
        let merged = merger(split);
        assert_eq!(original, merged);
    }

    #[test]
    fn halfer() {
        let u64 = [1, 2, 3, 4, 5, 6];
        let xm = &u64[..u64.len() / 2];
        let mx = &u64[u64.len() / 2..];
        assert_eq!(xm.len(), mx.len());
    }
    #[test]
    fn test_decompress_able_buffer() {
        let context = Arc::new(
            Context::initialize_with_adapter("NVIDIA".to_string())
                .block_on()
                .unwrap(),
        );
        for buffer_size in vec![256, 512, 1024, 2048].into_iter() {
            let filename = format!("{buffer_size}_chimp64_output.txt");
            if fs::exists(&filename).unwrap() {
                fs::remove_file(&filename).unwrap();
            }
            unsafe {
                set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
            }
            println!("Buffer size: {}", env::var("CHIMP_BUFFER_SIZE").unwrap());
            let mut messages = Vec::<String>::with_capacity(30);
            let values = get_values("city_temperature.csv")
                .expect("Could not read test values")
                .to_vec();
            for size_checkpoint in (1..11).progress() {
                let mut value_new = values[0..(values.len() * size_checkpoint) / 10].to_vec();
                log::info!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let compressor = ChimpCompressorBatched64::new(context.clone());
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
                let decompressor = decompressor::ChimpDecompressorBatched64::new(context.clone());
                match decompressor.decompress(&mut compressed_values2).block_on() {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();
                        messages.push(format!(
                            "Decoding time {size_checkpoint}0%: {decompression_time}\n"
                        ));
                        fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        fs::write("expected.log", value_new.iter().join("\n")).unwrap();
                        assert_eq!(decompressed_values, value_new);
                    }
                    Err(err) => {
                        eprintln!("Decompression error: {err:?}");
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
    fn test_decompress_able_64() {
        let subscriber = tracing_subscriber::fmt()
            .compact()
            .with_env_filter("wgpu_compress_64_batched=info")
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
        let context = Arc::new(
            Context::initialize_with_adapter("Intel".to_string())
                .block_on()
                .unwrap(),
        );
        for file_name in vec![
            // "city_temperature.csv",
            // "SSD_HDD_benchmarks.csv",
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
            let values = get_values(file_name)
                .expect("Could not read test values")
                .to_vec();
            for size_checkpoint in (8..9).progress() {
                let mut value_new = values[0..(values.len() * size_checkpoint) / 10].to_vec();

                let s = value_new
                    .iter()
                    .map(|it| format!("{:064b}", it.to_bits()))
                    .join("\n");
                fs::write("values", s).unwrap();

                log::info!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let compressor = ChimpCompressorBatched64::new(context.clone()); //.with_device(CPU);
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
                let decompressor = decompressor::GPUDecompressorBatched64::new(context.clone());
                match decompressor.decompress(&mut compressed_values2).block_on() {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();
                        messages.push(format!(
                            "Decoding time {size_checkpoint}0%: {decompression_time}\n"
                        ));
                        fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        fs::write("expected.log", value_new.iter().join("\n")).unwrap();
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

    fn get_third(field: &str) -> Option<String> {
        field
            .split(",")
            .collect_vec()
            .get(2)
            .map(|it| it.to_string())
    }
    //noinspection ALL
    fn get_values(file_name: impl Into<String>) -> anyhow::Result<Vec<f64>> {
        let dir = env::current_dir()?;
        let file_path = dir.parent().unwrap().join(file_name.into());
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

    #[test]
    fn test_bits() {
        let value = 83.80002877190708f64.to_bits();
        let value2 = 83.8f64.to_bits();
        println!("{value:064b}");
        println!("{value2:064b}");
    }
}
