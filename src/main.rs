use anyhow::Result;
use compress::cpu_compress::{
    CPUCompressor, Compressor, Decompressor, TimedCompressor, TimedDecompressor,
};
use compress::utils::general_utils::{check_for_debug_mode, open_file_for_append};
use itertools::Itertools;
use std::cmp::min;
use std::io::Write;
use std::{env, fs};

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::init();
    let mut values = get_values().expect("Could not read test values")[0..600].to_vec();

    // for (i, value) in values.iter().enumerate() {
    //     println!("{}:{}", i, value);
    // }

    //Scenario for gpu_compress
    gpu_compress(&mut values).await?;

    //Scenario for cpu_compress.await?;
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
    // if fs::exists("output_diff")? {
    //     fs::remove_file("output_diff")?;
    // }
    //
    // let mut file = open_file_for_append("output_diff").expect("Couldn't open file");
    // let mut diff_msg = String::new();
    // for i in 0..min(values.len(), decompressed.len()) {
    //     if values[i] != decompressed[i] {
    //         diff_msg.push_str(&format!(
    //             "difference at {i}: {} vs {}",
    //             values[i], decompressed[i]
    //         ));
    //         if diff_msg.lines().count() % 64 == 0 {
    //             write!(file, "{}", diff_msg).expect("Couldn't append to file");
    //             diff_msg = String::new();
    //         }
    //     }
    // }
    // write!(file, "{}", diff_msg).expect("Couldn't append to file");
    Ok(())
}

async fn gpu_compress(values: &mut Vec<f32>) -> Result<()> {
    println!("Starting compression of {} values", values.len());
    let mut compressor = compress::ChimpCompressor::default();
    if check_for_debug_mode().expect("Could not read file system") {
        compressor.set_debug(true);
    }
    let cpu_model = TimedDecompressor::from(CPUCompressor::default());

    let mut compressed = compressor.compress(values).await?;
    println!("Finished compression of {} values", values.len());

    println!("Started decompression");
    let decompressed = cpu_model.decompress(&mut compressed).await?;
    println!("Finished decompression");
    if fs::exists("output_diff")? {
        fs::remove_file("output_diff")?;
    }
    let mut file = open_file_for_append("output_diff").expect("Couldn't open file");
    let mut diff_msg = String::new();
    for i in 0..min(values.len(), decompressed.len()) {
        // if values[i] != decompressed[i] {
        diff_msg.push_str(&format!(
            "difference at {i}: {} vs {}\n",
            values[i], decompressed[i]
        ));
        if diff_msg.lines().count() % 64 == 0 {
            write!(file, "{}", diff_msg).expect("Couldn't append to file");
            diff_msg = String::new();
        }
        // }
    }
    write!(file, "{}", diff_msg).expect("Couldn't append to file");
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

fn get_third(field: &str) -> Option<String> {
    field
        .split(",")
        .collect_vec()
        .get(2)
        .map(|it| it.to_string())
}

#[cfg(test)]
mod compress_test {
    use crate::get_values;
    use compress::cpu_compress::{CPUCompressor, Compressor, Decompressor, TimedDecompressor};
    use compress::utils::general_utils::check_for_debug_mode;

    #[test]
    pub fn test_wgpu() {
        env_logger::init();

        let mut values = get_values().expect("Could not read test values");
        log::info!("Starting compression of {} values", values.len());
        let mut compressor = compress::ChimpCompressor::default();
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let cpu_model = TimedDecompressor::from(CPUCompressor::default());

        let mut compressed = pollster::block_on(compressor.compress(&mut values)).unwrap();
        log::info!("Finished compression of {} values", values.len());

        log::info!("Started decompression");
        let decompressed = pollster::block_on(cpu_model.decompress(&mut compressed)).unwrap();
        log::info!("Finished decompression");

        let mut count_equal = 0;
        let mut count_almost_equal = 0;
        decompressed.iter().zip(&values).for_each(|(a, b)| {
            if a == b {
                count_equal += 1;
            } else if f32::abs(a - b) < 0.01 {
                count_almost_equal += 1;
            }
        });
        log::info!(
            "The number of values that are equal to the initial dataset is {}({})",
            count_equal,
            100 * count_equal / values.len()
        );
        log::info!(
            "The number of values that are almost equal to the initial dataset is {}({})",
            count_almost_equal,
            100 * count_almost_equal / values.len()
        );

        assert_eq!(count_equal, values.len());
    }
}
