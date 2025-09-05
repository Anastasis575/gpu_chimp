use crate::cpu::finalize::extract_bits;
use crate::info;
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::bit_utils::BitReadable;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, DecompressResult, Step};
use compress_utils::time_it;
use itertools::Itertools;
use log::trace;
use std::cmp::{max, min};
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
            // let str_temp2 = format!("{:032b}", last_num);

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
                // let str_temp = format!("{:032b}", value);

                input_index += significant_bits as usize;
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
    #[allow(unused)]
    pub fn decompress_impl(
        &self,
        vec: &[u8],
    ) -> Result<DecompressResult<f32>, BatchedDecompressorError> {
        let mut current_index = 0usize;
        let mut output = Vec::new();
        let mut total_millis = 0;
        time_it!(
            {
                while current_index < vec.len() {
                    let buffer_size = u32::from_be_bytes(
                        vec[current_index..current_index + size_of::<u32>()]
                            .try_into()
                            .unwrap(),
                    ) as usize
                        + 1;
                    current_index += size_of::<u32>();

                    let size = u32::from_be_bytes(
                        vec[current_index..current_index + size_of::<u32>()]
                            .try_into()
                            .unwrap(),
                    );
                    current_index += size_of::<u32>();
                    let vec_view = vec[current_index..current_index + (size as usize)].to_vec();
                    let bit_vec = BitVec::from_bytes(&vec_view);
                    let block_values = self.decompress_block(&bit_vec)?;

                    output.extend(block_values[0..buffer_size].iter());
                    current_index += size as usize;
                }
            },
            total_millis,
            "decompression"
        );
        Ok(output.into())
    }
}

#[async_trait]
impl Decompressor<f32> for BatchedDecompressorCpu {
    async fn decompress(
        &self,
        vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f32>, DecompressionError> {
        self.decompress_impl(vec).map_err(DecompressionError::from)
    }
}

#[derive(Debug, Default, Clone)]
pub struct DebugBatchDecompressorCpu {}
#[async_trait]
impl Decompressor<f32> for DebugBatchDecompressorCpu {
    #[allow(unused)]
    async fn decompress(
        &self,
        compressed_bytes_vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f32>, DecompressionError> {
        let mut current_index = 0usize;
        let uncompressed_values;
        let mut total_millis = 0;
        time_it!(
            {
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < compressed_bytes_vec.len() {
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
                    let byte_window_vec = compressed_bytes_vec
                        [current_index..current_index + (size_in_bytes as usize)]
                        .to_vec();
                    let mut byte_window = byte_window_vec.as_slice();
                    assert_eq!(
                        byte_window.len() % 4,
                        0,
                        "Total bytes need to be in batches of 4"
                    );
                    while let Some((first_four_bytes, rest)) = byte_window.split_at_checked(4) {
                        byte_window = rest;
                        //parse u32 from groups of 4 bytes
                        let value_u32 = u32::from_be_bytes(first_four_bytes.try_into().unwrap());
                        vec_window.push(value_u32);
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
                        ChimpBufferInfo::get().buffer_size(),
                    )
                    .await?;

                uncompressed_values = block_values[0..total_uncompressed_values]
                    .iter()
                    .copied()
                    .collect_vec();
            },
            total_millis,
            "decompression"
        );
        Ok(uncompressed_values.into())
    }
}

impl DebugBatchDecompressorCpu {
    pub(crate) async fn decompress_block(
        &self,
        compressed_value_slice: &[u32],
        input_indexes: &[u32],
        buffer_value_count: usize,
    ) -> Result<Vec<f32>, DecompressionError> {
        // Number of buffers that fit into the GPU
        let workgroup_count = input_indexes.len() - 1;

        //how many iterations I need to fully decompress all the buffers
        let iterator_count = max((input_indexes.len() - 1) / workgroup_count, 1);

        //input_indexes shows how many buffers of count buffer_value_count, so we use workgroups equal to as many fit in the gpu
        let mut result = Vec::new();
        //info!("The wgpu workgroup size: {}", &workgroup_count);

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
                * size_of::<u32>();
            info!(
                "The uncompressed output values buffer size in bytes: {}",
                out_buffer_size
            );

            //info!("Total output values: {}", buffer_value_count);
            //info!("Total input values: {}", buffer_value_count);

            let workgroups = iteration_input_indexes.len() - 1;

            let output = vec![0f32; workgroups * ChimpBufferInfo::get().buffer_size()];
            let mut writer = CPUWrite::new(
                iteration_compressed_values,
                output,
                iteration_input_indexes,
                buffer_value_count as u32,
            );
            for workgroup in 0..workgroups {
                let index = writer.input_index[workgroup];

                writer.write(index, (workgroup * buffer_value_count).try_into().unwrap())
            }
            result.extend(writer.output());
        }
        //info!("Output result size: {}", result.len());
        if trace_steps().contains(&Step::Decompress) {
            let trace_path = Step::Decompress.get_trace_file();
            let mut trace_output = String::new();

            result
                .iter()
                .for_each(|it| trace_output.push_str(it.to_string().as_str()));

            fs::write(&trace_path, trace_output)
                .map_err(|it| DecompressionError::FromBaseAnyhowError(anyhow::anyhow!(it)))?;
        }
        Ok(result)
    }
}

struct CPUWrite {
    input: Vec<u32>,
    output: Vec<f32>,
    input_index: Vec<u32>,
    size: u32,
}

impl CPUWrite {
    fn output(self) -> Vec<f32> {
        self.output
    }
    pub(crate) fn write(&mut self, input_index: u32, output_index: u32) {
        let current_index = input_index + 1u32;
        //Current Remaining offset
        let current_offset = 0u32;

        let mut current_info = CurrentInfo::new(current_index, current_offset);

        let first_num = self.input[(current_info.current_index - 1) as usize];
        let mut last_num: u32 = first_num;
        let mut last_lead = 0u32;
        let mut significant_bits;

        let mut output_index = output_index;

        self.output[output_index as usize] = bytemuck::cast::<u32, f32>(first_num);
        let f_value = self.output[output_index as usize];
        trace!("{}:{:?}", output_index, f_value);
        output_index += 1u32;
        current_info.current_offset += 32u32;
        // current_info.current_index+=1;
        let mut value;
        for _i in 1..self.size {
            // if we have not finished reading values from the uncompressed buffers
            if current_info.current_index >= (self.input.len() as u32 - 1u32)
                && current_info.current_offset - 1u32 == 0
            {
                break;
            }

            //if current bit value==1
            if self.get_bit_at_index(current_info.current_index, current_info.current_offset) == 1 {
                current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                let mut lead = last_lead;
                if self.get_bit_at_index(current_info.current_index, current_info.current_offset)
                    == 1
                {
                    current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                    lead = self.reinterpret_num(
                        current_info.current_index,
                        current_info.current_offset,
                        5u32,
                    );
                    current_info = self.decr_counter_capped_at_32(current_info, 5u32);
                } else {
                    current_info = self.decr_counter_capped_at_32(current_info, 1u32);
                }
                significant_bits = 32u32 - lead;
                if significant_bits == 0u32 {
                    significant_bits = 32u32;
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

                self.output[output_index as usize] = bytemuck::cast::<u32, f32>(value);
                let f_value = self.output[output_index as usize];
                trace!("{}:{:?}", output_index, f_value);
                output_index += 1u32;
            } else if self.get_bit_at_index(
                current_info.current_index,
                current_info.current_offset - 1u32,
            ) == 1u32
            {
                current_info = self.decr_counter_capped_at_32(current_info, 2u32);
                let lead = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    5u32,
                );
                current_info = self.decr_counter_capped_at_32(current_info, 5u32);
                let mut significant_bits = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    5u32,
                );
                current_info = self.decr_counter_capped_at_32(current_info, 5u32);
                if significant_bits == 0u32 {
                    significant_bits = 32u32;
                }
                let trail = 32u32 - lead - significant_bits;
                value = self.reinterpret_num(
                    current_info.current_index,
                    current_info.current_offset,
                    32u32 - lead - trail,
                );
                current_info = self.decr_counter_capped_at_32(current_info, 32u32 - lead - trail);
                value <<= trail;
                value ^= last_num;
                last_lead = lead;
                last_num = value;

                self.output[output_index as usize] = bytemuck::cast::<u32, f32>(value);
                let f_value = self.output[output_index as usize];
                trace!("{}:{:?}", output_index, f_value);
                output_index += 1u32;
            } else {
                self.output[output_index as usize] = bytemuck::cast::<u32, f32>(last_num);
                let f_value = self.output[output_index as usize];
                trace!("{}:{:?}", output_index, f_value);
                output_index += 1u32;

                last_lead = 32u32;
                current_info = self.decr_counter_capped_at_32(current_info, 2u32);
            }
        }
    }
}

impl CPUWrite {
    pub fn new(input: Vec<u32>, output: Vec<f32>, input_index: Vec<u32>, size: u32) -> Self {
        Self {
            input,
            output,
            input_index,
            size,
        }
    }
    fn get_bit_at_index(&self, array_index: u32, position: u32) -> u32 {
        let mut index = array_index;
        let mut f_position = position;
        if f_position == 0 {
            index += 1;
            f_position = 32;
        }
        (self.input[index as usize] >> (f_position - 1)) & 1u32
    }

    fn decr_counter_capped_at_32(&self, mut value: CurrentInfo, count: u32) -> CurrentInfo {
        let corrected_value = value.current_offset as i32 - count as i32;
        value.current_offset = (if corrected_value > 0 { 1 } else { 0 } * (corrected_value)
            + if corrected_value <= 0 { 1 } else { 0 } * (32 + corrected_value))
            as u32;
        value.current_index += if corrected_value <= 0 { 1 } else { 0 }; //1 if it's true, and 0 otherwise
        value
    }

    fn reinterpret_num(&self, array_index: u32, index: u32, length: u32) -> u32 {
        let len = min(length, 32u32);
        if index >= len {
            // Fully within one u32
            extract_bits(self.input[array_index as usize], index - len, len)
        } else {
            // Spans two u32 elements
            let _bits_in_first = index;
            let bits_in_second = length - index;

            let first_part = extract_bits(self.input[array_index as usize], 0, index);
            let second_part = extract_bits(
                self.input[(array_index + 1) as usize],
                32u32 - bits_in_second,
                bits_in_second,
            );
            (first_part << bits_in_second) | second_part
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
