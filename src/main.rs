use anyhow::Result;
use itertools::Itertools;
use std::{env, fs};

#[tokio::main]
pub async fn main() -> Result<()> {
    env_logger::init();
    let dir = env::current_dir()?;
    let file_path = dir.join("city_temperature.csv");
    let file_txt = fs::read_to_string(file_path)?;
    let mut values = file_txt
        .split("\n")
        .map(get_third)
        .filter(|p| p.is_some())
        .map(|s| s.unwrap().parse::<f32>().unwrap())
        .collect_vec();

    let compressor=compress::ChimpCompressor::default();
    let compressed=compressor.chimp_compress(&mut values).await?;

    Ok(())
}


fn get_third(field: &str) -> Option<String> {
    field.split(",").collect_vec().get(2).map(|it|it.to_string())
}
