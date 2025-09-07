use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::{
    ChimpBufferInfo, DecompressResult, MaxGroupGnostic, Step, trace_steps,
};
use compress_utils::{
    BufferWrapper, WgpuGroupId, execute_compute_shader, step, time_it, wgpu_utils,
};
use itertools::Itertools;
use log::info;
use pollster::FutureExt;
use std::cmp::{max, min};
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use wgpu::{Device, Queue};
use wgpu_compress_32_batched::cpu::finalize::extract_bits;
use wgpu_compress_64_batched::ChimpCompressorBatched64;
use wgpu_compress_64_batched::cpu::utils_64;
use wgpu_types::BufferAddress;

#[async_trait]
impl Decompressor<f64> for BatchedCPUN64Decompressor {
    #[allow(unused)]
    async fn decompress(
        &self,
        compressed_bytes_vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f64>, DecompressionError> {
        let mut current_index = 0usize;
        let mut uncompressed_values = Vec::new();
        let mut total_millis = 0;
        let mut skip_time = 0;
        time_it!(
            {
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < compressed_bytes_vec.len() {
                    while current_index < compressed_bytes_vec.len() {
                        let old_index = current_index;
                        let buffer_value_count = u32::from_be_bytes(
                            compressed_bytes_vec[current_index..current_index + size_of::<u32>()]
                                .try_into()
                                .unwrap(),
                        ) as usize
                            + 1;
                        current_index += size_of::<u32>();

                        let size_in_bytes = u32::from_be_bytes(
                            compressed_bytes_vec[current_index..current_index + size_of::<u32>()]
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
                        let byte_window_vec = compressed_bytes_vec
                            [current_index..current_index + (size_in_bytes as usize)]
                            .to_vec();
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
                            &mut skip_time,
                        )
                        .await?;

                    uncompressed_values.extend(block_values[0..total_uncompressed_values].iter());
                }
            },
            total_millis,
            "decompression"
        );
        Ok(uncompressed_values.into())
    }
}

pub struct BatchedCPUN64Decompressor {
    context: Arc<Context>,
    n: usize,
}
impl MaxGroupGnostic for BatchedCPUN64Decompressor {
    fn get_max_number_of_groups(&self, _content_len: usize) -> usize {
        self.context().get_max_workgroup_size()
    }
}
impl Default for BatchedCPUN64Decompressor {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            n: 64,
        }
    }
}
impl BatchedCPUN64Decompressor {
    pub(crate) async fn decompress_block(
        &self,
        compressed_value_slice: &[u64],
        input_indexes: &[u32],
        buffer_value_count: usize,
        skip_time: &mut u128,
    ) -> Result<Vec<f64>, DecompressionError> {
        //how many buffers fit into the GPU
        let workgroup_count = self.get_max_number_of_groups(input_indexes.len());

        //how many iterations I need to fully decompress all the buffers
        let iterator_count = max((input_indexes.len() - 1) / workgroup_count, 1);

        //input_indexes shows how many buffers of count buffer_value_count, so we use workgroups equal to as many fit in the gpu
        let mut result = Vec::<f64>::new();
        //info!("The wgpu workgroup size: {}", &workgroup_count);

        let mut output =
            vec![0f64; (input_indexes.len() - 1) * ChimpBufferInfo::get().buffer_size()];
        let mut last_lead_array =
            vec![0u64; (input_indexes.len() - 1) * ChimpBufferInfo::get().buffer_size()];
        let mut writer = CPU64DecompressorNWriter {
            n: self.n,
            size: buffer_value_count,
            input_indexes: input_indexes.to_vec(),
            input: compressed_value_slice.to_vec(),
            input_size: input_indexes.len() - 1,
            output,
            last_lead_array,
        };
        let workgroup_count = min(workgroup_count, self.context.get_max_workgroup_size());
        for iteration in 0..iterator_count {
            let is_last_iteration = iteration == iterator_count - 1;
            let offset = iteration * workgroup_count;
            let next = if is_last_iteration {
                input_indexes.len()
            } else {
                ((iteration + 1) * workgroup_count) + 1
            };
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

            let size_uniform = BufferWrapper::uniform_with_content(
                self.device(),
                bytemuck::bytes_of(&buffer_value_count),
                WgpuGroupId::new(0, 2),
                Some("Total input values"),
            );

            let input_size_uniform = BufferWrapper::uniform_with_content(
                self.device(),
                bytemuck::bytes_of(&compressed_value_slice.len()),
                WgpuGroupId::new(0, 4),
                Some("Total input buffer length"),
            );

            for workgroup in 0..(iteration_input_indexes.len() - 1) {
                let offset = iteration * self.context.get_max_workgroup_size();
                writer.input_size = iteration_compressed_values.len();
                writer.write(
                    input_indexes[offset + workgroup] as usize,
                    (offset + workgroup) * buffer_value_count,
                )
            }
        }
        // let instant = Instant::now();
        // let result = wgpu_utils::get_from_gpu::<f32>(
        //     self.context(),
        //     out_storage_buffer.buffer(),
        //     out_storage_buffer.size() as BufferAddress,
        //     out_staging.buffer(),
        // )
        // .await?;
        // *skip_time += instant.elapsed().as_millis();

        //info!("Output result size: {}", result.len());
        Ok(writer.output)
    }

    pub fn new(context_builder: impl Into<Arc<Context>>, n: usize) -> Self {
        Self {
            context: context_builder.into(),
            n,
        }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn device(&self) -> &Device {
        self.context.device()
    }
    pub fn queue(&self) -> &Queue {
        self.context.queue()
    }
}

pub struct CPU64DecompressorNWriter {
    input: Vec<u64>,
    n: usize,
    output: Vec<f64>,
    input_indexes: Vec<u32>,
    last_lead_array: Vec<u64>,
    size: usize,
    input_size: usize,
}
impl CPU64DecompressorNWriter {
    pub fn write(&mut self, input_idx: usize, output_idx: usize) {
        let n = self.n;
        let log2n = self.n.ilog2();

        //Index of the byte we are in
        let mut current_index = input_idx + 1usize;
        //Current Remaining offset
        let mut current_offset = 0u32;

        let mut current_info = CurrentInfo::new(current_index as u32, current_offset);

        let mut first_num = self.input[(current_info.current_index - 1u32) as usize];
        let mut last_num: u64 = first_num;
        let mut last_lead = 0u32;
        let mut significant_bits = 0u32;

        let mut output_index = output_idx;

        self.output[output_index] = bytemuck::cast(first_num);
        output_index += 1usize;
        current_info.current_offset += 32u32;

        let mut value;
        for i in 1..self.size {
            // if we have not finished reading values from the uncompressed buffers
            if current_info.current_index >= (self.input_size as u32 - 1u32)
                && current_info.current_offset <= 1u32
            {
                break;
            }

            //if current bit value==1
            if self.get_bit_at_index(current_info.current_index, current_info.current_offset)
                == 1u32
            {
                current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                let recalc_lead = self
                    .get_bit_at_index(current_info.current_index, current_info.current_offset)
                    == 1;
                current_info = self.decr_counter_capped_at_32(current_info, 1u32);

                let compare_offset = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    log2n,
                ) as usize;
                current_info = self.decr_counter_capped_at_32(current_info, log2n);
                let mut last_num = bytemuck::cast(self.output[output_index - compare_offset]);
                let mut lead = self.last_lead_array[output_index - compare_offset];
                if recalc_lead {
                    lead = self.reinterpret_num(
                        current_info.current_index,
                        current_info.current_offset,
                        6u32,
                    );
                    current_info = self.decr_counter_capped_at_32(current_info, 6u32);
                }
                significant_bits = 64u32 - (lead as u32);
                if significant_bits == 0u32 {
                    significant_bits = 64u32;
                }
                value = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    (significant_bits) as u32,
                );
                current_info =
                    self.decr_counter_capped_at_32(current_info, (significant_bits) as u32);
                value = value ^ last_num;
                last_num = value;
                self.last_lead_array[output_index] = lead;

                self.output[output_index] = bytemuck::cast(value);
                output_index += 1usize;
            } else if self.get_bit_at_index(
                current_info.current_index,
                current_info.current_offset - 1u32,
            ) == 1u32
            {
                current_info = self.decr_counter_capped_at_32(current_info, 2u32);
                let compare_offset = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    log2n,
                ) as usize;
                current_info = self.decr_counter_capped_at_32(current_info, log2n);
                let mut last_num = bytemuck::cast(self.output[output_index - compare_offset]);

                let lead = self.reinterpret_num(
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
                self.last_lead_array[output_index] = lead;
                last_num = value;

                self.output[output_index] = bytemuck::cast(value);
                output_index += 1usize;
            } else {
                current_info = self.decr_counter_capped_at_32(current_info, 2u32);
                let compare_offset = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    log2n,
                ) as usize;
                current_info = self.decr_counter_capped_at_32(current_info, log2n);
                let mut last_num: u32 = bytemuck::cast(self.output[output_index - compare_offset]);
                let mut lead = self.last_lead_array[output_index - compare_offset];
                self.output[output_index] = bytemuck::cast(last_num);
                self.last_lead_array[output_index] = 64u64;
                output_index += 1usize;
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
        ((self.input[index as usize] >> (f_position - 1u32)) & 1u64) as u32
    }

    fn decr_counter_capped_at_32(&self, mut value: CurrentInfo, count: u32) -> CurrentInfo {
        let corrected_value = (value.current_offset as i32) - count as i32;
        (value).current_offset = if (corrected_value > 0) {
            corrected_value as u32
        } else {
            (64 + corrected_value) as u32
        };
        (value).current_index += if (corrected_value <= 0) { 1 } else { 0 }; //1 if it's true and 0 otherwise
        return (value);
    }

    fn reinterpret_num(&self, array_index: u32, index: u32, length: u32) -> u64 {
        let len = min(length, 64u32);
        if index >= len {
            // Fully within one u64
            return utils_64::extract_bits(
                self.input[array_index as usize],
                (index - len) as u32,
                len,
            );
        } else {
            // Spans two u32 elements
            let bits_in_first = index;
            let bits_in_second = length - index;

            let first_part = utils_64::extract_bits(self.input[array_index as usize], 0u32, index);
            let second_part = utils_64::extract_bits(
                self.input[array_index as usize + 1],
                64u32 - bits_in_second,
                bits_in_second,
            );
            return (first_part << bits_in_second) | second_part;
        }
    }
}

struct CurrentInfo {
    current_index: u32,
    current_offset: u32,
}

impl CurrentInfo {
    pub fn new(current_index: u32, current_offset: u32) -> Self {
        Self {
            current_index,
            current_offset,
        }
    }
}
