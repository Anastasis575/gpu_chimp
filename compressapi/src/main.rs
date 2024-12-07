use anyhow::Result;
use compress_utils::bit_utils::to_bit_vec;
use compress_utils::cpu_compress::{
    CPUCompressor, Compressor, Decompressor, TimedCompressor, TimedDecompressor,
};
use compress_utils::general_utils::check_for_debug_mode;
use dotenv::dotenv;
use itertools::Itertools;
use log::info;
use std::{env, fs};

#[tokio::main]
pub async fn main() -> Result<()> {
    // dotenv().expect("Failed to read .env file");
    env_logger::init();
    let mut values = get_values().expect("Could not read test values")[..256].to_vec();

    if check_for_debug_mode().expect("Could not read file system") {
        for (i, value) in values.iter().enumerate() {
            info!("{}:{} - {}", i, value, to_bit_vec(value.to_bits()));
        }
    }

    //Scenario for gpu_compress
    gpu_compress_batched(&mut values).await?;
    gpu_compress(&mut values).await?;

    //Scenario for cpu compression
    // cpu_compress(&mut values).await?;
    Ok(())
}

//noinspection DuplicatedCode
async fn cpu_compress(values: &mut Vec<f32>) -> Result<()> {
    let mut cpu_model = CPUCompressor::new(false);
    let timed_compressor = TimedCompressor::from(cpu_model.clone());
    let timed_decompressor = TimedDecompressor::from(cpu_model.clone());
    if check_for_debug_mode().expect("Could not read file system") {
        cpu_model.set_debug(true);
    }

    println!("Starting compression of {} values", values.len());
    let mut compressed = timed_compressor.compress(values).await?;
    println!("Finished compression of {} values", values.len());

    println!("Started decompression");
    let decompressed = timed_decompressor.decompress(&mut compressed).await?;
    println!("Finished decompression");
    count_matching_values(values, &decompressed);
    Ok(())
}

async fn gpu_compress_batched(values: &mut Vec<f32>) -> Result<()> {
    println!("Starting batched compression of {} values", values.len());
    let mut compressor = wgpu_compress_32_batched::ChimpCompressorBatched::default();
    if check_for_debug_mode().expect("Could not read file system") {
        compressor.set_debug(true);
    }
    let cpu_model = TimedDecompressor::from(CPUCompressor::default());

    let mut compressed = compressor.compress(values).await?;
    println!("Finished compression of {} values", values.len());

    // println!("Started decompression");
    // let decompressed = cpu_model.decompress(&mut compressed).await?;
    // println!("Finished decompression");
    // count_matching_values(values, &decompressed);
    Ok(())
}
async fn gpu_compress(values: &mut Vec<f32>) -> Result<()> {
    println!("Starting compression of {} values", values.len());
    let mut compressor = wgpu_compress_32::ChimpCompressor::default();
    if check_for_debug_mode().expect("Could not read file system") {
        compressor.set_debug(true);
    }
    let cpu_model = TimedDecompressor::from(CPUCompressor::default());

    let mut compressed = compressor.compress(values).await?;
    println!("Finished compression of {} values", values.len());

    // println!("Started decompression");
    // let decompressed = cpu_model.decompress(&mut compressed).await?;
    // println!("Finished decompression");
    // count_matching_values(values, &decompressed);
    Ok(())
}

fn get_values() -> Result<Vec<f32>> {
    let dir = env::current_dir()?;
    let file_path = dir.join("city_temperature.csv");
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

fn count_matching_values(values: &Vec<f32>, decompressed: &[f32]) -> usize {
    let mut count_equal = 0;
    let mut count_almost_equal = 0;
    decompressed.iter().zip(values).for_each(|(a, b)| {
        if a == b {
            count_equal += 1;
        } else if f32::abs(a - b) < 0.01 {
            count_almost_equal += 1;
        }
    });
    info!(
        "The number of values that are equal to the initial dataset is {}({})",
        count_equal,
        100 * count_equal / values.len()
    );
    info!(
        "The number of values that are almost equal to the initial dataset is {}({})",
        count_almost_equal,
        100 * count_almost_equal / values.len()
    );
    count_equal
}

fn get_third(field: &str) -> Option<String> {
    field
        .split(",")
        .collect_vec()
        .get(2)
        .map(|it| it.to_string())
}

#[cfg(test)]
mod compress_test {
    use crate::{count_matching_values, get_values};
    use compress_utils::cpu_compress::{
        CPUCompressor, Compressor, Decompressor, TimedDecompressor,
    };
    use compress_utils::general_utils::check_for_debug_mode;

    #[test]
    pub fn wgpu_compress_32() {
        env_logger::init();

        let mut values = get_values().expect("Could not read test values");
        log::info!("Starting compression of {} values", values.len());
        let mut compressor =
            wgpu_compress_32::ChimpCompressor::new("NVIDIA".to_string(), false).unwrap();
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let cpu_model = TimedDecompressor::from(CPUCompressor::default());

        let mut compressed = pollster::block_on(compressor.compress(&mut values)).unwrap();
        log::info!("Finished compression of {} values", values.len());
        // let total_compressed_bytes = compressed.len() * size_of::<u8>();
        log::info!("Started decompression");
        let decompressed = pollster::block_on(cpu_model.decompress(&mut compressed)).unwrap();
        log::info!("Finished decompression");

        let count_equal = count_matching_values(&values, &decompressed);
        log::info!(
            "The total size of the compressed dataset is {}",
            values.len() * size_of::<f32>(),
        );
        log::info!(
            "The mean bit count per number is {}%",
            100 * compressed.len() / (values.len() * size_of::<f32>()),
        );

        assert_eq!(count_equal, values.len());
    }
}
