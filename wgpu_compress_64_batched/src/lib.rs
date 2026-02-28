use crate::calculate_indexes::{CalculateIndexes64, GPUCalculateIndexes64};
use crate::compute_s_shader::{ComputeS, ComputeSImpl};
use crate::final_compress::{FinalCompress, FinalCompressImpl64};
use crate::finalize::{Finalize, Finalizer64};
use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::{
    ChimpBufferInfo, CompressResult, DeviceEnum, MaxGroupGnostic, Padding,
};
use compress_utils::types::{ChimpOutput64, S};
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{time_it, wgpu_utils};
use itertools::Itertools;
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
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<CompressResult, CompressionError> {
        let compute_s_impl = self.compute_s_factory();
        let final_compress_impl = self.compute_final_compress_factory();
        let indexes_impl = self.calculate_index_factory();
        let finalize_impl = self.compute_finalize_factory();

        let mut iterations = self.split_by_max_gpu_buffer_size(vec);
        let mut byte_stream = Vec::new();
        let mut metadata = 0;
        let mut skip_time = 0;
        for iteration_values in iterations {
            let mut total_millis = 0;
            let mut values = iteration_values;
            let mut output_vec: CompressResult;
            let mut s_values: Vec<S>;
            let mut chimp_vec: Vec<ChimpOutput64>;
            let mut indexes: Vec<u32>;
            let mut padding = Padding(0);
            let buffer_size = ChimpBufferInfo::get().buffer_size();
            let mut buffers = wgpu_utils::RunBuffers::default();
            values = add_padding_to_fit_buffer_count_64(values, buffer_size, &mut padding);
            time_it!(
                {
                    compute_s_impl
                        .compute_s(&mut values, &mut buffers, &mut skip_time)
                        .await?;
                },
                total_millis,
                "s computation stage"
            );
            time_it!(
                {
                    final_compress_impl
                        .final_compress(&mut buffers, &mut skip_time)
                        .await?;
                },
                total_millis,
                "compression stage"
            );
            time_it!(
                {
                    indexes_impl
                        .calculate_indexes(
                            &mut buffers,
                            ChimpBufferInfo::get().buffer_size() as u32,
                            &mut skip_time,
                        )
                        .await?;
                },
                total_millis,
                "calculate trim size stage"
            );
            time_it!(
                {
                    output_vec = finalize_impl
                        .finalize(&mut buffers, padding.0, &mut skip_time)
                        .await?;
                },
                total_millis,
                "trimming stage"
            );
            byte_stream.extend(output_vec.compressed_value_ref());
            metadata += output_vec.metadata_size();
        }
        Ok(CompressResult(byte_stream, metadata, skip_time))
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
    async fn compute_s(
        &self,
        values: &mut [f64],
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> anyhow::Result<()> {
        match self {
            ComputeS64Impls::GPU(c) => c.compute_s(values, buffers, skip_time).await,
            ComputeS64Impls::CPU(c) => {
                c.compute_s(values, buffers, skip_time).await?;
                Ok(())
            }
        }
    }
}

enum CalculateIndexesImpls {
    GPU(GPUCalculateIndexes64),
    CPU(cpu::calculate_indexes::CPUCalculateIndexes64),
}
#[async_trait]
impl CalculateIndexes64 for CalculateIndexesImpls {
    async fn calculate_indexes(
        &self,
        buffers: &mut RunBuffers,
        size: u32,
        skip_time: &mut u128,
    ) -> Result<()> {
        match self {
            CalculateIndexesImpls::GPU(c) => c.calculate_indexes(buffers, size, skip_time).await,
            CalculateIndexesImpls::CPU(c) => {
                // c.calculate_indexes(input, size).await?;
                Ok(())
            }
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
        run_buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> anyhow::Result<()> {
        match self {
            Compress64Impls::GPU(c) => c.final_compress(run_buffers, skip_time).await,
            Compress64Impls::CPU(c) => {
                // c.final_compress(input, s_values, padding).await;
                Ok(())
            }
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
        run_buffers: &mut RunBuffers,
        padding: usize,
        skip_time: &mut u128,
    ) -> anyhow::Result<CompressResult> {
        match self {
            Finalizer64impls::GPU(f) => f.finalize(run_buffers, padding, skip_time).await,
            Finalizer64impls::CPU(f) => {
                // f.finalize(chimp_output, padding, indexes).await?;
                Ok(CompressResult(Vec::new(), 0, 0))
            }
        }
    }
}
#[allow(unused)]
impl ChimpCompressorBatched64 {
    pub const MAX_BUFFER_SIZE_BYTES: usize = 134217728;
    fn compute_s_factory(&self) -> ComputeS64Impls {
        match self.device_type() {
            DeviceEnum::GPU => ComputeS64Impls::GPU(ComputeSImpl::new(self.context.clone())),
            DeviceEnum::CPU => {
                ComputeS64Impls::CPU(cpu::compute_s::CpuComputeSImpl::new(self.context.clone()))
            }
        }
    }
    fn split_by_max_gpu_buffer_size(&self, vec: &mut Vec<f64>) -> Vec<Vec<f64>> {
        let max = self.context.get_max_storage_buffer_size();
        let mut split_by = max / size_of::<S>() - ChimpBufferInfo::get().buffer_size(); //The most costly buffer
        while ((split_by + 10) * size_of::<S>()) as u64
            >= self.context.get_max_storage_buffer_size() as u64
            || ((split_by + 10) * size_of::<ChimpOutput64>()) as u64
                >= self.context.get_max_storage_buffer_size() as u64
        {
            split_by -= ChimpBufferInfo::get().buffer_size();
        }
        let closest = split_by - split_by % ChimpBufferInfo::get().buffer_size();

        let x = vec.chunks(closest).map(|it| it.to_vec()).collect_vec();
        x
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
    use compress_utils::general_utils::{build_event_times, EventLogType};
    use env::set_var;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::cmp::min;
    use std::sync::Arc;
    use std::{env, fs};
    use tracing_subscriber::fmt::MakeWriter;

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
        // let subscriber = tracing_subscriber::fmt()
        //     .compact()
        //     .with_env_filter("wgpu_compress_64_batched=info")
        //     // .with_writer(
        //     //     OpenOptions::new()
        //     //         .create(true)
        //     //         .truncate(true)
        //     //         .write(true)
        //     //         .open("run.log")
        //     //         .unwrap(),
        //     // )
        //     .finish();
        // subscriber.init();
        let context = Arc::new(
            Context::initialize_with_adapter("NVIDIA".to_string())
                .block_on()
                .unwrap(),
        );
        for buffer_size in vec![256, 512, 1024, 2048].into_iter() {
            let filename_no_io = format!("{buffer_size}_chimp64_output_no_io.txt");
            if fs::exists(&filename_no_io).unwrap() {
                fs::remove_file(&filename_no_io).unwrap();
            }
            // let filename = format!("{buffer_size}_chimp64_output.txt");
            // if fs::exists(&filename).unwrap() {
            //     fs::remove_file(&filename).unwrap();
            // }
            unsafe {
                set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
            }
            println!("Buffer size: {}", env::var("CHIMP_BUFFER_SIZE").unwrap());
            // let mut messages = Vec::<EventLogType>::with_capacity(30);
            let mut messages_no_io = Vec::<EventLogType>::with_capacity(30);
            let mut values = get_values("city_temperature.csv")
                .expect("Could not read test values")
                .to_vec();
            let mut reader = TimeSeriesReader::new(50_000, values.clone(), 500_000_000);
            for size_checkpoint in 1..11 {
                while let Some(block) = reader.next() {
                    values.extend(block);
                    if values.len() >= (size_checkpoint * reader.max_size()) / 100 {
                        break;
                    }
                }
                let mut value_new = values.clone();
                println!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let compressor = ChimpCompressorBatched64::new(context.clone());
                let mut compressed_values2 =
                    compressor.compress(&mut value_new).block_on().unwrap();
                let compression_time = time.elapsed().as_millis();

                const SIZE_IN_BYTE: usize = 8;
                let compression_ratio = (compressed_values2.compressed_value_ref().len()
                    * SIZE_IN_BYTE) as f64
                    / value_new.len() as f64;
                // messages.push(EventLogType::CompressionRatio {values:value_new.len() as u64, ratio:compression_ratio});
                messages_no_io.push(EventLogType::CompressionRatio {
                    values: value_new.len() as u64,
                    ratio: compression_ratio,
                });

                // messages.push(EventLogType::EncodingTime {values:value_new.len() as u64, time:compression_time});
                messages_no_io.push(EventLogType::EncodingTime {
                    values: value_new.len() as u64,
                    time: compression_time - compressed_values2.skip_time(),
                });

                let time = std::time::Instant::now();
                let decompressor = decompressor::ChimpDecompressorBatched64::new(context.clone());
                match decompressor
                    .decompress(compressed_values2.compressed_value_mut())
                    .block_on()
                {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();

                        // messages.push(EventLogType::DecompressionTime {values:value_new.len() as u64, time:decompression_time});
                        messages_no_io.push(EventLogType::DecompressionTime {
                            values: value_new.len() as u64,
                            time: decompression_time - decompressed_values.skip_time(),
                        });

                        assert_eq!(decompressed_values.un_compressed_value(), value_new);
                    }
                    Err(err) => {
                        eprintln!("Decompression error: {err:?}");
                        panic!("{}", err);
                    }
                }
            }

            let mut writer = csv::Writer::from_path(filename_no_io).unwrap();

            let logs = build_event_times(messages_no_io);
            for message in logs.iter().sorted_by_key(|it| it.values) {
                writer.serialize(message).unwrap();
            }
        }
    }
    #[test]
    fn test_decompress_able_64() {
        // let subscriber = tracing_subscriber::fmt()
        //     .compact()
        //     .with_env_filter("wgpu_compress_64_batched=info")
        //     // .with_writer(
        //     //     OpenOptions::new()
        //     //         .create(true)
        //     //         .truncate(true)
        //     //         .write(true)
        //     //         .open("run.log")
        //     //         .unwrap(),
        //     // )
        //     .finish();
        // subscriber.init();
        let adapter = env::var("CHIMP_GPU_ADAPTER").unwrap_or("NVIDIA".to_string());
        let context = Arc::new(
            Context::initialize_with_adapter(adapter)
                .block_on()
                .unwrap(),
        );
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
            let filename_no_io = format!("{}_chimp64_output_no_io.txt", &file_name);
            if fs::exists(&filename_no_io).unwrap() {
                fs::remove_file(&filename_no_io).unwrap();
            }
            let mut messages = Vec::<EventLogType>::with_capacity(30);
            let mut messages_no_io = Vec::<EventLogType>::with_capacity(30);
            let mut values = get_values(file_name)
                .expect("Could not read test values")
                .to_vec();
            let mut reader = TimeSeriesReader::new(50_000, values.clone(), 500_000_000);
            for size_checkpoint in 1..11 {
                while let Some(block) = reader.next() {
                    values.extend(block);
                    if values.len() >= (size_checkpoint * reader.max_size()) / 100 {
                        break;
                    }
                }
                let mut value_new = values.clone();

                println!("Starting compression of {} values", value_new.len());
                let time = std::time::Instant::now();
                let compressor = ChimpCompressorBatched64::new(context.clone()); //.with_device(CPU);
                let mut compressed_values2 =
                    compressor.compress(&mut value_new).block_on().unwrap();
                let compression_time = time.elapsed().as_millis();

                const SIZE_IN_BYTE: usize = 8;
                let compression_ratio = (compressed_values2.compressed_value_ref().len()
                    * SIZE_IN_BYTE) as f64
                    / value_new.len() as f64;
                messages.push(EventLogType::CompressionRatio {
                    values: value_new.len() as u64,
                    ratio: compression_ratio,
                });
                messages_no_io.push(EventLogType::CompressionRatio {
                    values: value_new.len() as u64,
                    ratio: compression_ratio,
                });

                messages.push(EventLogType::EncodingTime {
                    values: value_new.len() as u64,
                    time: compression_time,
                });
                messages_no_io.push(EventLogType::EncodingTime {
                    values: value_new.len() as u64,
                    time: compression_time - compressed_values2.skip_time(),
                });

                let time = std::time::Instant::now();
                let decompressor = decompressor::GPUDecompressorBatched64::new(context.clone());
                match decompressor
                    .decompress(compressed_values2.compressed_value_mut())
                    .block_on()
                {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();

                        messages.push(EventLogType::DecompressionTime {
                            values: value_new.len() as u64,
                            time: decompression_time,
                        });
                        messages_no_io.push(EventLogType::DecompressionTime {
                            values: value_new.len() as u64,
                            time: decompression_time - decompressed_values.skip_time(),
                        });
                        // fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        // fs::write("expected.log", value_new.iter().join("\n")).unwrap();
                        assert_eq!(decompressed_values.un_compressed_value(), value_new);
                    }
                    Err(err) => {
                        eprintln!("Decompression error: {:?}", err);
                        panic!("{}", err);
                    }
                }
            }
            let mut writer = csv::Writer::from_path(filename).unwrap();

            let logs = build_event_times(messages);
            for message in logs.iter().sorted_by_key(|it| it.values) {
                writer.serialize(message).unwrap();
            }
            let mut writer = csv::Writer::from_path(filename_no_io).unwrap();

            let logs = build_event_times(messages_no_io);
            for message in logs.iter().sorted_by_key(|it| it.values) {
                writer.serialize(message).unwrap();
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

    struct TimeSeriesReader {
        minimum_block_size: usize,
        block_size: usize,
        current_block: usize,
        source_value: Vec<f64>,
        current_index: usize,
    }

    impl TimeSeriesReader {
        pub fn new(block_size: usize, source_value: Vec<f64>, minimum_block_size: usize) -> Self {
            Self {
                minimum_block_size,
                block_size: min(block_size, source_value.len()),
                current_block: 0,
                source_value,
                current_index: 0,
            }
        }
        pub fn max_size(&self) -> usize {
            let len = self.source_value.len();
            let mut max = len;
            while max < self.minimum_block_size {
                max += self.block_size;
            }
            max
        }
    }

    impl Iterator for TimeSeriesReader {
        type Item = Vec<f64>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut block = Vec::<f64>::with_capacity(self.block_size);
            if self.current_index <= self.minimum_block_size {
                if self.block_size + self.current_index > self.source_value.len() {
                    let remaining = self.block_size + self.current_index - self.source_value.len();
                    block.extend(&self.source_value[self.current_index..]);
                    block.extend(&self.source_value[0..remaining]);
                    self.current_index = remaining - 1;
                    Some(block)
                } else {
                    block.extend(
                        &self.source_value
                            [self.current_index..self.current_index + self.block_size],
                    );
                    self.current_index += self.block_size;
                    Some(block)
                }
            } else {
                None
            }
        }
    }

    //noinspection ALL
    fn get_values(file_name: impl Into<String>) -> anyhow::Result<Vec<f64>> {
        let dir = env::current_dir()?;
        let file_path = dir.parent().unwrap().join(file_name.into());
        let file_txt = fs::read_to_string(file_path)?;
        let mut values = Vec::new();
        values.extend(
            file_txt
                .split("\n")
                .map(get_third)
                .filter(|p| p.is_some())
                .map(|s| s.unwrap().parse::<f64>().unwrap())
                .into_iter(),
        );
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
