use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::bit_utils::BitReadable;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use itertools::Itertools;
use std::fs;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BatchedDecompressorError {
    #[error("Invalid Decompression Format")]
    Default,
}

impl From<BatchedDecompressorError> for DecompressionError {
    fn from(value: BatchedDecompressorError) -> Self {
        DecompressionError::FromBaseAnyhowError(anyhow::Error::from(value))
    }
}

#[derive(Debug, Default, Clone)]
pub struct BatchedDecompressorCpu {}

impl BatchedDecompressorCpu {
    pub fn decompress_block(
        &self,
        input_vector: &BitVec,
    ) -> Result<Vec<f32>, BatchedDecompressorError> {
        let mut input_index: usize;
        let first_num_u32: u32 = input_vector.reinterpret_u32(0, 32);
        let first_num = f32::from_bits(first_num_u32);
        // if self.debug {
        //     log::info!("0:{}", first_num);
        // }

        let mut output = vec![first_num];

        let mut last_num = first_num.to_bits();
        let mut last_lead = 0;
        input_index = 32;
        while input_index < input_vector.len() {
            if input_index + 1 >= input_vector.len() {
                break;
            }
            if input_vector[input_index] {
                input_index += 1;
                let mut lead = last_lead;
                if input_vector[input_index] {
                    input_index += 1;
                    lead = input_vector.reinterpret_u32(input_index, 5);
                    input_index += 5;
                } else {
                    input_index += 1;
                }
                if lead > 32 {
                    return Err(BatchedDecompressorError::Default);
                }
                let mut significant_bits = 32 - lead;
                if significant_bits == 0 {
                    significant_bits = 32;
                }
                let value = input_vector.reinterpret_u32(input_index, significant_bits as usize);
                input_index += (32 - lead) as usize;
                let value = value ^ last_num;
                last_num = value;
                last_lead = lead;

                if value == u32::MAX {
                    break;
                } else {
                    let value_f32 = f32::from_bits(value);
                    // if self.debug {
                    //     log::info!("{}:{}", output.len(), value_f32);
                    // }
                    output.push(value_f32);
                }
            } else if input_vector[input_index + 1] {
                input_index += 2;
                let lead = input_vector.reinterpret_u32(input_index, 5);
                input_index += 5;
                let mut significant_bits = input_vector.reinterpret_u32(input_index, 5);
                input_index += 5;
                if significant_bits == 0 {
                    significant_bits = 32;
                }
                if lead + significant_bits > 32 {
                    return Err(BatchedDecompressorError::Default);
                }
                let trail = 32 - lead - significant_bits;
                let mut value =
                    input_vector.reinterpret_u32(input_index, (32 - lead - trail) as usize);
                input_index += (32 - lead - trail) as usize;
                value <<= trail;
                value ^= last_num;
                last_lead = lead;
                last_num = value;
                if value == u32::MAX {
                    break;
                } else {
                    let value_f32 = f32::from_bits(value);
                    // if self.debug {
                    //     log::info!("{}:{}", output.len(), value_f32);
                    // }
                    output.push(value_f32);
                }
            } else {
                let value_f32 = f32::from_bits(last_num);
                last_lead = 32;
                // if self.debug {
                //     log::info!("{}:{}", output.len(), value_f32);
                // }
                output.push(value_f32);
                input_index += 2;
            }
        }
        Ok(output)
    }
    pub fn decompress_impl(&self, vec: &[u8]) -> Result<Vec<f32>, BatchedDecompressorError> {
        let mut current_index = 0usize;
        let mut output = Vec::new();
        let byte_count_type_size = size_of::<u32>();

        let mut i = 0;

        while current_index < vec.len() {
            let buffer_size = u32::from_be_bytes(
                vec[current_index..current_index + byte_count_type_size]
                    .try_into()
                    .unwrap(),
            ) as usize;
            current_index += byte_count_type_size;
            let size = u32::from_be_bytes(
                vec[current_index..current_index + byte_count_type_size]
                    .try_into()
                    .unwrap(),
            );
            current_index += byte_count_type_size;
            let vec_view = vec[current_index..current_index + (size as usize)].to_vec();
            fs::write(
                format!("output_work_{i}.log"),
                vec_view.iter().map(|it| format!("{:08b}", it)).join("\n"),
            )
            .unwrap();
            i += 1;
            let bit_vec = BitVec::from_bytes(&vec_view);
            let block_values = self.decompress_block(&bit_vec)?;

            output.extend(block_values[0..buffer_size].iter());
            current_index += size as usize;
        }
        Ok(output)
    }
}
#[async_trait]
impl Decompressor for BatchedDecompressorCpu {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>, DecompressionError> {
        self.decompress_impl(vec).map_err(DecompressionError::from)
    }
}
