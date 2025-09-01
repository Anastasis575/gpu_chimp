mod calculate_indexes;
mod compute_s_shader;
pub mod cpu;
pub mod decompressor;
mod final_compress;
mod finalize;

use crate::calculate_indexes::{CalculateIndexes, GPUCalculateIndexes};
use crate::compute_s_shader::{ComputeS, ComputeSImpl};
use crate::cpu::finalize::CPUImpl;
use crate::final_compress::{FinalCompress, FinalCompressImpl};
use crate::finalize::{Finalize, Finalizer};
use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::CompressResult;
pub use compress_utils::general_utils::{
    add_padding_to_fit_buffer_count, ChimpBufferInfo, DeviceEnum, Padding,
};
use compress_utils::types::{ChimpOutput, S};
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{time_it, wgpu_utils};
use itertools::Itertools;
use log::info;
use pollster::FutureExt;
use std::sync::Arc;

#[derive(Debug)]
pub enum FinalizerImpl {
    GPU(Finalizer),
    CPU(CPUImpl),
}

#[async_trait]
impl Finalize for FinalizerImpl {
    async fn finalize(
        &self,
        run_buffers: &mut RunBuffers,
        padding: usize,
        skip_time: &mut u128,
    ) -> Result<CompressResult> {
        match self {
            FinalizerImpl::GPU(impll) => impll.finalize(run_buffers, padding, skip_time).await,
            FinalizerImpl::CPU(impll) => Ok(CompressResult(Vec::new(), 0, 0)), //impll.finalize(chimp_output, padding, indexes).await,
        }
    }
}
#[derive(Debug)]
pub struct ChimpCompressorBatched {
    debug: bool,
    context: Arc<Context>,
    finalizer: DeviceEnum,
}
impl Default for ChimpCompressorBatched {
    fn default() -> Self {
        Self {
            debug: false,
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            finalizer: DeviceEnum::GPU,
        }
    }
}

#[async_trait]
impl Compressor<f32> for ChimpCompressorBatched {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<CompressResult, CompressionError> {
        let compute_s_impl = self.compute_s_factory();
        let final_compress_impl = self.compute_final_compress_factory();
        let calculate_indexes_impl = self.calculate_indexes_factory();
        let finalize_impl = self.compute_finalize_factory();

        let iterations = self.split_by_max_gpu_buffer_size(vec);
        let mut byte_stream = Vec::new();
        let mut metadata = 0usize;
        let mut buffers = wgpu_utils::RunBuffers::default();
        let mut skip_time = 0u128;
        for iteration_values in iterations {
            let mut padding = Padding(0);
            let buffer_size = ChimpBufferInfo::get().buffer_size();
            let mut values = iteration_values;
            values = add_padding_to_fit_buffer_count(values, buffer_size, &mut padding);
            let mut total_millis: u128 = 0;
            let mut s_values: Vec<S>;
            let mut chimp_vec: Vec<ChimpOutput>;
            // let mut indexes;
            let output_vec;
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
                "final output stage"
            );
            time_it!(
                {
                    calculate_indexes_impl
                        .calculate_indexes(
                            &mut buffers,
                            ChimpBufferInfo::get().buffer_size() as u32,
                            &mut skip_time,
                        )
                        .await?;
                },
                total_millis,
                "final output stage"
            );
            time_it!(
                {
                    output_vec = finalize_impl
                        .finalize(&mut buffers, padding.0, &mut skip_time)
                        .await?;
                },
                total_millis,
                "final Result collection"
            );
            byte_stream.extend(output_vec.compressed_value_ref());
            metadata += output_vec.metadata_size()
        }

        Ok(CompressResult(byte_stream, metadata, skip_time))
    }
}

impl ChimpCompressorBatched {
    pub const MAX_BUFFER_SIZE_BYTES: usize = 134_217_728;

    pub fn new(debug: bool, context: Arc<Context>, finalizer: DeviceEnum) -> Self {
        Self {
            debug,
            context,
            finalizer,
        }
    }

    fn split_by_max_gpu_buffer_size(&self, vec: &mut Vec<f32>) -> Vec<Vec<f32>> {
        let max = self.context.get_max_storage_buffer_size();
        let mut split_by = max / size_of::<S>() - ChimpBufferInfo::get().buffer_size(); //The most costly buffer
        while ((split_by + 10) * size_of::<S>()) as u64
            >= self.context.get_max_storage_buffer_size() as u64
            || ((split_by + 10) * size_of::<ChimpOutput>()) as u64
                >= self.context.get_max_storage_buffer_size() as u64
        {
            split_by -= ChimpBufferInfo::get().buffer_size();
        }
        let closest = split_by - split_by % ChimpBufferInfo::get().buffer_size();

        let x = vec.chunks(closest).map(|it| it.to_vec()).collect_vec();
        x
    }
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub fn debug(&self) -> bool {
        self.debug
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
    fn compute_s_factory(&self) -> impl ComputeS {
        ComputeSImpl::new(self.context().clone())
    }
    fn compute_final_compress_factory(&self) -> impl FinalCompress + use<'_> {
        FinalCompressImpl::new(self.context().clone(), self.debug())
    }
    fn calculate_indexes_factory(&self) -> impl CalculateIndexes {
        GPUCalculateIndexes::new(self.context().clone())
    }
    fn compute_finalize_factory(&self) -> impl Finalize + use<'_> {
        match self.finalizer {
            DeviceEnum::GPU => FinalizerImpl::GPU(Finalizer::new(self.context().clone())),
            DeviceEnum::CPU => FinalizerImpl::CPU(CPUImpl::default()),
        }
    }
}
#[cfg(test)]
mod tests {
    use crate::decompressor::BatchedGPUDecompressor;
    use crate::ChimpCompressorBatched;
    use crate::DeviceEnum::GPU;
    use compress_utils::context::Context;
    use compress_utils::cpu_compress::{Compressor, Decompressor};
    use compress_utils::general_utils::check_for_debug_mode;
    use indicatif::ProgressIterator;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::cmp::min;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;
    use std::{env, fs};
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
        // let subscriber = tracing_subscriber::fmt()
        //     .compact()
        //     .with_env_filter("wgpu_compress_32=info")
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

        let mut values = get_values("city_temperature.csv")
            .expect("Could not read test values")
            .to_vec();
        // log:: //info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched::default();
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let compressed_values1 = compressor.compress(&mut values).block_on().unwrap();

        let mut values = get_values("city_temperature.csv")
            .expect("Could not read test values")
            .to_vec();
        // log:: //info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched {
            finalizer: GPU,
            ..ChimpCompressorBatched::default()
        };
        // if check_for_debug_mode().expect("Could not read file system") {
        //     compressor.set_debug(true);
        // }
        let compressed_values2 = compressor.compress(&mut values).block_on().unwrap();
        assert_eq!(
            compressed_values2.compressed_value(),
            compressed_values1.compressed_value()
        );
    }
    //noinspection DuplicatedCode
    #[test]
    fn test_decompress_able() {
        // let subscriber = tracing_subscriber::fmt()
        //     .compact()
        //     .with_env_filter("wgpu_compress_32=info")
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
        env::set_var("CHIMP_BUFFER_SIZE", "1024".to_string());

        for file_name in vec![
            "city_temperature.csv",
            "SSD_HDD_benchmarks.csv",
            "Stocks-Germany-sample.txt",
        ]
        .into_iter()
        {
            println!("{file_name}");
            let filename = format!("{}_chimp32_output_no_io.txt", &file_name);
            if fs::exists(&filename).unwrap() {
                fs::remove_file(&filename).unwrap();
            }
            let mut messages = Vec::<String>::with_capacity(30);
            let mut values = get_values(file_name)
                .expect("Could not read test values")
                .to_vec();
            let mut reader = TimeSeriesReader::new(500_000, values.clone(), 500_000_000);
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
                let mut compressor = ChimpCompressorBatched::new(false, context.clone(), GPU);
                let mut compressed_values2 =
                    compressor.compress(&mut value_new).block_on().unwrap();
                let compression_time = time.elapsed().as_millis();

                const SIZE_IN_BYTE: usize = 8;
                let compression_ratio = (compressed_values2.compressed_value_ref().len()
                    * SIZE_IN_BYTE) as f64
                    / value_new.len() as f64;
                messages.push(format!(
                    "Compression ratio {} values: {compression_ratio}\n",
                    value_new.len()
                ));
                messages.push(format!(
                    "Encoding time {} values: {}\n",
                    value_new.len(),
                    compression_time - compressed_values2.skip_time()
                ));

                let time = std::time::Instant::now();
                let decompressor = BatchedGPUDecompressor::new(context.clone());
                match decompressor
                    .decompress(compressed_values2.compressed_value_mut())
                    .block_on()
                {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();
                        messages.push(format!(
                            "Decoding time {} values: {}\n",
                            value_new.len(),
                            decompression_time - decompressed_values.skip_time()
                        ));
                        // fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        // fs::write("expected.log", value_new.iter().join("\n")).unwrap();
                        assert_eq!(decompressed_values.0, value_new);
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
    fn test_decompress_able_buffer() {
        // let subscriber = tracing_subscriber::fmt()
        //     .compact()
        //     .with_env_filter("wgpu_compress_32=info")
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
            let filename = format!("{}_chimp32_output.txt", buffer_size);
            if fs::exists(&filename).unwrap() {
                fs::remove_file(&filename).unwrap();
            }
            env::set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
            println!("Buffer size: {}", env::var("CHIMP_BUFFER_SIZE").unwrap());
            let mut messages = Vec::<String>::with_capacity(30);
            let mut values = get_values("city_temperature.csv")
                .expect("Could not read test values")
                .to_vec();

            let mut reader = TimeSeriesReader::new(50_000, values.clone(), 500_000_000);

            for size_checkpoint in (1..11) {
                while let Some(block) = reader.next() {
                    values.extend(block);
                    if values.len() >= (size_checkpoint * reader.max_size()) / 100 {
                        break;
                    }
                }
                let mut value_new = values.clone();
                println!("Starting compression of {} values", value_new.len());
                let time = std::time::Instant::now();
                let mut compressor = ChimpCompressorBatched::new(false, context.clone(), GPU);
                let mut compressed_values2 =
                    compressor.compress(&mut value_new).block_on().unwrap();
                let compression_time = time.elapsed().as_millis();

                const SIZE_IN_BYTE: usize = 8;
                let compression_ratio = (compressed_values2.compressed_value_ref().len()
                    * SIZE_IN_BYTE) as f64
                    / value_new.len() as f64;
                messages.push(format!(
                    "Compression ratio {} values: {compression_ratio}\n",
                    value_new.len()
                ));
                messages.push(format!(
                    "Encoding time {} values: {compression_time}\n",
                    value_new.len()
                ));

                let time = std::time::Instant::now();
                let decompressor = BatchedGPUDecompressor::new(context.clone());
                match decompressor
                    .decompress(&mut compressed_values2.compressed_value_mut())
                    .block_on()
                {
                    Ok(decompressed_values) => {
                        let decompression_time = time.elapsed().as_millis();
                        messages.push(format!(
                            "Decoding time {} values: {decompression_time}\n",
                            value_new.len()
                        ));
                        // fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                        // fs::write("expected.log", values.iter().join("\n")).unwrap();
                        assert_eq!(decompressed_values.0, value_new);
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
        // log:: //info!("Starting compression of {} values", values.len());
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
        match decompressor
            .decompress(&mut compressed_values2.compressed_value_mut())
            .block_on()
        {
            Ok(decompressed_values) => {
                // fs::write("actual.log", decompressed_values.iter().join("\n")).unwrap();
                // fs::write("expected.log", values.iter().join("\n")).unwrap();
                assert_eq!(decompressed_values.0, values)
            }
            Err(err) => {
                eprintln!("Decompression error: {:?}", err);
                panic!("{}", err);
            }
        }
    }
    struct TimeSeriesReader {
        minimum_block_size: usize,
        block_size: usize,
        current_block: usize,
        source_value: Vec<f32>,
        current_index: usize,
    }

    impl TimeSeriesReader {
        pub fn new(block_size: usize, source_value: Vec<f32>, minimum_block_size: usize) -> Self {
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
        type Item = Vec<f32>;

        fn next(&mut self) -> Option<Self::Item> {
            let mut block = Vec::<f32>::with_capacity(self.block_size);
            if self.current_index < self.minimum_block_size {
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
}
