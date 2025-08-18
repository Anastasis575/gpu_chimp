use crate::cpu::utils_64;
use crate::ChimpCompressorBatched64;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::{ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::{step, time_it};
use itertools::Itertools;
use log::info;
use std::cmp::{max, min};
use std::fs;
use std::sync::Arc;

pub struct CPUDecompressorBatched64 {
    context: Arc<Context>,
}
impl MaxGroupGnostic for CPUDecompressorBatched64 {
    fn get_max_number_of_groups(&self, _content_len: usize) -> usize {
        self.context().get_max_workgroup_size()
    }
}
#[async_trait]
impl Decompressor<f64> for CPUDecompressorBatched64 {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
        let mut current_index = 0usize;
        let mut uncompressed_values = Vec::new();
        let mut total_millis = 0;
        time_it!(
            {
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < vec.len() {
                    while current_index < vec.len() {
                        let old_index = current_index;
                        let buffer_value_count = u32::from_be_bytes(
                            vec[current_index..current_index + size_of::<u32>()]
                                .try_into()
                                .unwrap(),
                        ) as usize
                            + 1;
                        current_index += size_of::<u32>();

                        let size_in_bytes = u32::from_be_bytes(
                            vec[current_index..current_index + size_of::<u32>()]
                                .try_into()
                                .unwrap(),
                        );
                        current_index += size_of::<u32>();
                        if vec_window.len() + size_in_bytes as usize
                            >= ChimpCompressorBatched64::MAX_BUFFER_SIZE_BYTES
                        {
                            current_index = old_index;
                            break;
                        }
                        let byte_window_vec =
                            vec[current_index..current_index + (size_in_bytes as usize)].to_vec();
                        let mut byte_window = byte_window_vec.as_slice();
                        assert_eq!(
                            byte_window.len() % 8,
                            0,
                            "Total bytes need to be in batches of 8"
                        );
                        while let Some((first_four_bytes, rest)) = byte_window.split_at_checked(8) {
                            byte_window = rest;
                            //parse u32 from groups of 8 bytes
                            let value_u64 =
                                u64::from_le_bytes(first_four_bytes.try_into().unwrap());
                            vec_window.push(value_u64);
                        }
                        input_indexes.push(vec_window.len() as u32);
                        current_index += size_in_bytes as usize;
                        total_uncompressed_values += buffer_value_count
                    }
                    input_indexes.insert(0, 0);
                    //Block is as many buffers fit into the gpu the distinction is made for compatibility reasons
                    let block_values = self
                        .decompress_block(
                            vec_window.as_slice(),
                            input_indexes.as_slice(),
                            min(
                                total_uncompressed_values,
                                ChimpBufferInfo::get().buffer_size(),
                            ),
                        )
                        .await?;

                    uncompressed_values.extend(block_values[0..total_uncompressed_values].iter());
                }
            },
            total_millis,
            "decompression"
        );
        Ok(uncompressed_values)
    }
}
impl CPUDecompressorBatched64 {
    //noinspection DuplicatedCode
    pub(crate) async fn decompress_block(
        &self,
        compressed_value_slice: &[u64],
        input_indexes: &[u32],
        buffer_value_count: usize,
    ) -> Result<Vec<f64>, DecompressionError> {
        //how many buffers fit into the GPU
        let workgroup_count = self.get_max_number_of_groups(input_indexes.len());

        //how many iterations I need to fully decompress all the buffers
        let iterator_count = max((input_indexes.len() - 1) / workgroup_count, 1);

        //input_indexes shows how many buffers of count buffer_value_count, so we use workgroups equal to as many fit in the gpu
        let mut result = Vec::<f64>::new();
        info!("The wgpu workgroup size: {}", &workgroup_count);

        for iteration in 0..iterator_count {
            //split all the buffers to the chunks each iteration will use
            let is_last_iteration = iteration == iterator_count - 1;
            let iteration_input_indexes = if is_last_iteration {
                input_indexes[iteration * workgroup_count..].to_vec()
            } else {
                input_indexes[iteration * workgroup_count..(iteration + 1) * workgroup_count]
                    .to_vec()
            };
            let first_index = iteration_input_indexes[0] as usize;
            let iteration_compressed_values = if is_last_iteration {
                compressed_value_slice[first_index..].to_vec()
            } else {
                compressed_value_slice
                    [first_index..(iteration_input_indexes.last().unwrap().to_owned() as usize)]
                    .to_vec()
            };

            info!(
                "The size in bytes of the compressed input vec: {}",
                iteration_compressed_values.len() * size_of::<u8>()
            );

            let out_buffer_size = (iteration_input_indexes.len() - 1)
                * ChimpBufferInfo::get().buffer_size()
                * size_of::<f64>();
            info!(
                "The uncompressed output values buffer size in bytes: {}",
                out_buffer_size
            );
            let out_vec = vec![
                f64::default();
                (iteration_input_indexes.len() - 1)
                    * ChimpBufferInfo::get().buffer_size()
            ];

            info!("Total output values: {}", buffer_value_count);
            info!("Total input values: {}", buffer_value_count);

            let in_size = iteration_compressed_values.len();
            let iteration_workgroup_count = iteration_input_indexes.len() - 1;
            let mut writer = CPUDecompressWriter64 {
                input_index_vec: iteration_input_indexes,
                in_vec: iteration_compressed_values,
                input_size: in_size as u32,
                size: buffer_value_count as u32,
                out_vec,
            };
            for workgroup in 0..iteration_workgroup_count {
                writer.write(
                    input_indexes[workgroup],
                    (workgroup * buffer_value_count) as u32,
                )
            }
            result.extend(writer.out_vec);
        }
        info!("Output result size: {}", result.len());
        step!(Step::Decompress, {
            result.iter().map(|it| it.to_string()).into_iter()
        });
        Ok(result)
    }

    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn new(context: impl Into<Arc<Context>>) -> Self {
        Self {
            context: context.into(),
        }
    }
}

struct CPUDecompressWriter64 {
    out_vec: Vec<f64>,
    in_vec: Vec<u64>,
    size: u32,
    input_index_vec: Vec<u32>,
    input_size: u32,
}
struct CurrentInfo {
    current_index: u32,
    current_offset: u32,
}
impl CPUDecompressWriter64 {
    fn write(&mut self, input_idx: u32, output_idx: u32) {
        //Index of the byte we are in
        let current_index = input_idx + 1u32;
        //Current Remaining offset
        let current_offset = 0u32;

        let mut current_info = CurrentInfo {
            current_index,
            current_offset,
        };

        let first_num = self.in_vec[(current_info.current_index - 1u32) as usize];
        let mut last_num: u64 = first_num;
        let mut last_lead = 0u64;
        let mut significant_bits;

        let mut output_index = output_idx;

        self.out_vec[output_index as usize] = bytemuck::cast(first_num);
        output_index += 1u32;
        current_info.current_offset += 64u32;
        // current_info.current_index+=1;
        let mut value;
        for _i in 1..self.size {
            // if we have not finished reading values from the uncompressed buffers
            if current_info.current_index >= (self.input_size - 1u32)
                && (current_info.current_offset - 1u32) <= 0u32
            {
                break;
            }

            //if current bit value==1
            if self.get_bit_at_index(current_info.current_index, current_info.current_offset)
                == 1u32
            {
                current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                let mut lead = last_lead;
                if self.get_bit_at_index(current_info.current_index, current_info.current_offset)
                    == 1
                {
                    current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                    lead = self.reinterpret_num(
                        current_info.current_index,
                        current_info.current_offset,
                        6u32,
                    );
                    current_info = self.decr_counter_capped_at_32(current_info, 6u32);
                } else {
                    current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                }
                significant_bits = 64u32 - (lead as u32);
                if significant_bits == 0u32 {
                    significant_bits = 64u32;
                }
                value = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    significant_bits,
                );
                current_info = self.decr_counter_capped_at_32(current_info, significant_bits);
                value = value ^ last_num;
                last_num = value;
                last_lead = lead;

                self.out_vec[output_index as usize] = bytemuck::cast(value);
                output_index += 1u32;
            } else if self.get_bit_at_index(
                current_info.current_index,
                current_info.current_offset - 1u32,
            ) == 1u32
            {
                current_info = self.decr_counter_capped_at_32(current_info, 2u32);

                let lead: u64 = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    6u32,
                );
                current_info = self.decr_counter_capped_at_32(current_info, 6u32);

                let mut significant_bits = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    6u32,
                );
                current_info = self.decr_counter_capped_at_32(current_info, 6u32);

                if significant_bits == 0u64 {
                    significant_bits = 64;
                }

                let trail = 64u32 - (lead as u32) - (significant_bits as u32);

                value = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    64u32 - (lead as u32) - trail,
                );

                current_info =
                    self.decr_counter_capped_at_32(current_info, 64u32 - (lead as u32) - trail);

                value <<= trail;
                value ^= last_num;
                last_lead = lead;
                last_num = value;

                self.out_vec[output_index as usize] = bytemuck::cast(value);
                output_index += 1u32;
            } else {
                self.out_vec[output_index as usize] = bytemuck::cast(last_num);
                output_index += 1u32;

                last_lead = 64u64;
                current_info = self.decr_counter_capped_at_32(current_info, 2u32);
            }
        }
    }

    fn get_bit_at_index(&self, array_index: u32, position: u32) -> u32 {
        let index = if position == 0u32 {
            array_index + 1
        } else {
            array_index
        };
        let f_position = if position == 0u32 { 64u32 } else { position };
        ((self.in_vec[index as usize] >> (f_position - 1u32)) & 1u64) as u32
    }

    fn decr_counter_capped_at_32(&self, mut value: CurrentInfo, count: u32) -> CurrentInfo {
        let corrected_value = value.current_offset as i32 - (count as i32);
        value.current_offset = if corrected_value > 0 {
            corrected_value as u32
        } else {
            (64 + corrected_value) as u32
        };
        value.current_index += (corrected_value <= 0) as u32;
        value
    }

    fn reinterpret_num(&self, array_index: u32, index: u32, length: u32) -> u64 {
        let len = min(length, 64u32);
        if index >= len {
            // Fully within one u64
            utils_64::extract_bits(self.in_vec[array_index as usize], index - len, len)
        } else {
            // Spans two u64 elements
            let bits_in_second = length - index;

            let first_part = utils_64::extract_bits(self.in_vec[array_index as usize], 0u32, index);
            let second_part = utils_64::extract_bits(
                self.in_vec[(array_index + 1) as usize],
                64u32 - bits_in_second,
                bits_in_second,
            );
            (first_part << bits_in_second) | second_part
        }
    }
}
