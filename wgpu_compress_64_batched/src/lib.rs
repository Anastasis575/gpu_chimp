use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{
    CompressionError, Compressor, DecompressionError, Decompressor,
};
use itertools::Itertools;
use std::sync::Arc;
use wgpu_compress_32_batched::decompressor::BatchedGPUDecompressor;
use wgpu_compress_32_batched::{ChimpCompressorBatched, FinalizerEnum};
pub mod actual64;
mod compute_s_shader;
pub mod decompressor;
mod final_compress;
mod finalize;

#[derive(Debug)]
pub struct ChimpCompressorBatched64<T>
where
    T: Compressor<f32>,
{
    compressor32bit: T,
}

impl ChimpCompressorBatched64<ChimpCompressorBatched> {
    pub fn new(debug: bool, context: Arc<Context>, finalizer: FinalizerEnum) -> Self {
        Self {
            compressor32bit: ChimpCompressorBatched::new(debug, context, finalizer),
        }
    }
}
impl Default for ChimpCompressorBatched64<ChimpCompressorBatched> {
    fn default() -> Self {
        Self {
            compressor32bit: ChimpCompressorBatched::default(),
        }
    }
}

// fn splitter(value: f64) -> [f32; 2] {
//     const LOW_MASK: u64 = 0xFFFF_FFFF;
//     const SHIFT: u32 = 32;
//
//     let bits = bytemuck::cast::<f64, u64>(value);
//     let high_bits = (bits >> SHIFT) as u32;
//     let low_bits = (bits & LOW_MASK) as u32;
//     [high_bits as f32, low_bits as f32]
// }
// fn merger(value: [f32; 2]) -> f64 {
//     let high = value[0] as u64;
//     let low = value[1] as u64;
//     let bits = (high << 32) | low;
//     bytemuck::cast::<u64, f64>(bits)
// }

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

#[async_trait]
impl<T: Compressor<f32> + Send + Sync> Compressor<f64> for ChimpCompressorBatched64<T> {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError> {
        let mut split = vec.iter().map(|it| splitter(*it)).collect_vec();
        let mut final_values = split.iter_mut().map(|x| x[0]).collect_vec();
        let split_right_side = split.iter_mut().map(|x| x[1]).collect_vec();
        final_values.extend(split_right_side);
        let compressed = self.compressor32bit.compress(&mut final_values).await?;
        Ok(compressed)
    }
}

#[derive(Debug)]
pub struct ChimpDecompressorBatched64<T>
where
    T: Decompressor<f32>,
{
    decompressor32bits: T,
}

impl ChimpDecompressorBatched64<BatchedGPUDecompressor> {
    pub fn new(context: Arc<Context>) -> Self {
        Self {
            decompressor32bits: BatchedGPUDecompressor::new(context),
        }
    }
}
impl Default for ChimpDecompressorBatched64<BatchedGPUDecompressor> {
    fn default() -> Self {
        Self {
            decompressor32bits: BatchedGPUDecompressor::default(),
        }
    }
}

#[async_trait]
impl<T: Decompressor<f32> + Send + Sync> Decompressor<f64> for ChimpDecompressorBatched64<T> {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
        let decompressed = self.decompressor32bits.decompress(vec).await?;
        assert_eq!(decompressed.len() % 2, 0);
        let mut merged = Vec::with_capacity(decompressed.len() / 2 + 1);
        let half = decompressed.len() / 2;
        (0..half).for_each(|i| {
            let values = [decompressed[i], decompressed[half + i]];
            merged.push(merger(values));
        });
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        actual64, decompressor, merger, splitter, ChimpCompressorBatched64,
        ChimpDecompressorBatched64,
    };
    use compress_utils::context::Context;
    use compress_utils::cpu_compress::{Compressor, Decompressor};
    use env::set_var;
    use indicatif::ProgressIterator;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Arc;
    use std::{env, fs};
    use tracing_subscriber::fmt::MakeWriter;
    use wgpu_compress_32_batched::cpu;
    use wgpu_compress_32_batched::cpu::decompressor::DebugBatchDecompressorCpu;
    use wgpu_compress_32_batched::FinalizerEnum::GPU;

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
        for buffer_size in vec![256].into_iter() {
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
            // for size_checkpoint in (1..11).progress() {
            for size_checkpoint in (2..3).progress() {
                let mut value_new = values[0..(values.len() * size_checkpoint) / 10].to_vec();
                log::info!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let compressor = ChimpCompressorBatched64::new(false, context.clone(), GPU);
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
                let decompressor = ChimpDecompressorBatched64 {
                    decompressor32bits: DebugBatchDecompressorCpu::default(),
                };
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
        for file_name in vec![
            // "city_temperature.csv",
            "SSD_HDD_benchmarks.csv",
            // "Stocks-Germany-sample.txt", Problematic
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
            for size_checkpoint in (1..11).progress() {
                let mut value_new = values[0..(values.len() * size_checkpoint) / 10].to_vec();
                log::info!("Starting compression of {} values", values.len());
                let time = std::time::Instant::now();
                let compressor = actual64::ChimpCompressorBatched64::new(context.clone());
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
                let decompressor = decompressor::BatchedGPUDecompressor::new(context.clone());
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
