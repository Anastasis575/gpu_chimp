pub mod calculate_indexes;
pub mod chimpn;
pub mod compute_s_shader;
pub mod cpu;
pub mod decompressor;
pub mod final_compress;
pub mod finalize;
pub mod previous_indexes;

#[cfg(test)]
mod tests {
    use crate::chimpn::ChimpN64GPUBatched;
    use crate::decompressor::GPUDecompressorBatchedN64;
    use compress_utils::context::Context;
    use compress_utils::cpu_compress::{Compressor, Decompressor};
    use compress_utils::general_utils::EventLogType::DecompressionTime;
    use compress_utils::general_utils::{build_event_times, EventLogType};
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::cmp::min;
    use std::sync::Arc;
    use std::{env, fs};
    use tracing_subscriber::fmt::MakeWriter;

    #[test]
    fn test_decompress_able() {
        // let subscriber = tracing_subscriber::fmt()
        //     .compact()
        //     .with_env_filter("wgpu_compress_32_n_batched=info")
        //     //     // .with_writer(
        //     //     //     OpenOptions::new()
        //     //     //         .create(true)
        //     //     //         .truncate(true)
        //     //     //         .write(true)
        //     //     //         .open("run.log")
        //     //     //         .unwrap(),
        //     //     // )
        //     .finish();
        // subscriber.init();
        let context = Arc::new(
            Context::initialize_with_adapter("NVIDIA".to_string())
                .block_on()
                .unwrap(),
        );
        unsafe {
            env::set_var("CHIMP_BUFFER_SIZE", "1024".to_string());
        }
        for n in [32, 64, 128] {
            for file_name in vec![
                "city_temperature.csv",
                "SSD_HDD_benchmarks.csv",
                "Stocks-Germany-sample.txt",
            ]
            .into_iter()
            {
                println!("{file_name}");
                let filename = format!("{}_chimp64_n_{n}_output_no_io.txt", &file_name);
                if fs::exists(&filename).unwrap() {
                    fs::remove_file(&filename).unwrap();
                }
                let mut messages = Vec::<EventLogType>::with_capacity(30);
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
                    let mut value_new = values.to_vec();
                    println!("Starting compression of {} values", value_new.len());
                    let time = std::time::Instant::now();
                    let compressor = ChimpN64GPUBatched::new(context.clone(), n);
                    let mut compressed_values2 =
                        compressor.compress(&mut value_new).block_on().unwrap();
                    let compression_time = time.elapsed().as_millis();
                    // println!("{}", compression_time);
                    const SIZE_IN_BYTE: usize = 8;
                    let compression_ratio = (compressed_values2.compressed_value_ref().len()
                        * SIZE_IN_BYTE) as f64
                        / value_new.len() as f64;

                    messages.push(EventLogType::CompressionRatio {
                        values: value_new.len() as u64,
                        ratio: compression_ratio,
                    });
                    // println!("{}", messages.last().unwrap());

                    messages.push(EventLogType::EncodingTime {
                        values: value_new.len() as u64,
                        time: compression_time - compressed_values2.skip_time(),
                    });

                    // println!("{}", compression_time - compressed_values2.skip_time());
                    // println!("{}", messages.last().unwrap());

                    let time = std::time::Instant::now();
                    let decompressor = GPUDecompressorBatchedN64::new(context.clone(), n);
                    match decompressor
                        .decompress(compressed_values2.compressed_value_mut())
                        .block_on()
                    {
                        Ok(decompressed_values) => {
                            let decompression_time = time.elapsed().as_millis();

                            messages.push(DecompressionTime {
                                values: value_new.len() as u64,
                                time: decompression_time - decompressed_values.skip_time(),
                            });
                            // println!("{}", messages.last().unwrap());
                            // fs::write(
                            //     "actual.log",
                            //     decompressed_values
                            //         .un_compressed_value_ref()
                            //         .iter()
                            //         .join("\n"),
                            // )
                            // .unwrap();
                            // fs::write("expected.log", value_new.iter().join("\n")).unwrap();
                            assert_eq!(decompressed_values.0, value_new);
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
            }
        }
        // assert!(true)
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
}
