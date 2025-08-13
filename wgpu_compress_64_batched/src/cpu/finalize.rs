use crate::cpu::utils_64::{extract_bits, insert_bits};
use crate::finalize::Finalize;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::{ChimpBufferInfo, Step};
use compress_utils::step;
use compress_utils::types::ChimpOutput64;
use itertools::Itertools;
use std::cmp::{max, min};
use std::fs;
use std::ops::Div;
use std::sync::Arc;

#[derive(Debug)]
pub struct CPUFinalizer64(Arc<Context>);

impl CPUFinalizer64 {
    pub fn new(context: Arc<Context>) -> Self {
        Self(context)
    }
    pub fn context(&self) -> &Context {
        &self.0
    }
}
#[async_trait]
impl Finalize for CPUFinalizer64 {
    async fn finalize(
        &self,
        chimp_output: &mut Vec<ChimpOutput64>,
        padding: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let chimp_input_length = chimp_output.len() - padding;
        let _input_length = chimp_input_length;
        let output_vec = vec![0u64; chimp_output.len()];
        let workgroup_count = chimp_output.len().div(ChimpBufferInfo::get().buffer_size());
        let indexes = vec![0u32; workgroup_count];
        let size = ChimpBufferInfo::get().buffer_size();
        let mut writer =
            CPUWriter64::new(chimp_output.to_owned(), output_vec, size as u32, indexes);
        for workgroup in 0..max(workgroup_count, 1) {
            writer.last_byte_index[workgroup] = writer.write((workgroup * size) as u32)
        }

        let mut final_vec = Vec::<u8>::new();
        for (i, useful_byte_count) in writer.last_byte_index.iter().enumerate() {
            let start_index = i * ChimpBufferInfo::get().buffer_size();
            let byte_count = min(*useful_byte_count as usize, chimp_input_length - 1);
            let temp_vec = writer.out_vec[start_index..=byte_count]
                .iter()
                .flat_map(|it| it.to_be_bytes())
                .collect_vec();

            let batch_size = if i == workgroup_count - 1
                && chimp_input_length % ChimpBufferInfo::get().buffer_size() != 0
            {
                ((chimp_input_length % ChimpBufferInfo::get().buffer_size()) - 1) as u32
            } else {
                (ChimpBufferInfo::get().buffer_size() - 1) as u32
            };
            final_vec.extend(batch_size.to_be_bytes());
            final_vec.extend((temp_vec.len() as u32).to_be_bytes().iter());
            final_vec.extend(temp_vec);
        }
        step!(&Step::Finalize, {
            final_vec
                .iter()
                .map(|it| format!("{:08b}\n", it))
                .into_iter()
        });
        Ok(final_vec)
    }
}

pub struct CPUWriter64 {
    in_vec: Vec<ChimpOutput64>,
    out_vec: Vec<u64>,
    size: u32,
    last_byte_index: Vec<u32>,
}

impl CPUWriter64 {
    pub fn new(
        in_vec: Vec<ChimpOutput64>,
        out_vec: Vec<u64>,
        size: u32,
        last_byte_index: Vec<u32>,
    ) -> Self {
        Self {
            in_vec,
            out_vec,
            size,
            last_byte_index,
        }
    }

    fn get_fitting(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
        min(bits_rest_to_write, writeable_output_remaining)
    }

    fn get_remaining(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
        return max(
            bits_rest_to_write
                - CPUWriter64::get_fitting(bits_rest_to_write, writeable_output_remaining),
            0u32,
        );
    }

    fn get_insert_index(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
        return max(
            writeable_output_remaining
                - CPUWriter64::get_fitting(bits_rest_to_write, writeable_output_remaining),
            0u32,
        );
    }
    fn write(&mut self, idx: u32) -> u32 {
        let mut current_i = idx + 1u32;
        let mut current_i_bits_left = 64u32;

        self.out_vec[idx as usize] = self.in_vec[idx as usize].lower_bits;
        for i in idx + 1u32..idx + self.size {
            let chimp: ChimpOutput64 = self.in_vec[i as usize];
            let overflow_bits = (chimp.bit_count as i32) - 64;

            let mut fitting: u32;
            let mut insert_index: u32;
            let mut remaining: u32;

            let mut bits_to_add: u64;

            let rest_bits: u32;

            if overflow_bits > 0 {
                fitting = CPUWriter64::get_fitting(overflow_bits as u32, current_i_bits_left);
                insert_index =
                    CPUWriter64::get_insert_index(overflow_bits as u32, current_i_bits_left);
                remaining = CPUWriter64::get_remaining(overflow_bits as u32, current_i_bits_left);

                bits_to_add = extract_bits(
                    chimp.upper_bits,
                    (overflow_bits - (fitting as i32)) as u32,
                    fitting,
                );
                self.out_vec[current_i as usize] = insert_bits(
                    self.out_vec[current_i as usize],
                    bits_to_add,
                    insert_index,
                    fitting,
                );

                if current_i_bits_left <= fitting {
                    current_i += 1u32;
                    current_i_bits_left = 64u32;
                } else {
                    current_i_bits_left -= fitting;
                }
                if remaining > 0 {
                    fitting = CPUWriter64::get_fitting(remaining, current_i_bits_left);
                    insert_index = CPUWriter64::get_insert_index(remaining, current_i_bits_left);

                    bits_to_add = extract_bits(chimp.upper_bits, 0u32, fitting);
                    self.out_vec[current_i as usize] = insert_bits(
                        self.out_vec[current_i as usize],
                        bits_to_add,
                        insert_index,
                        fitting,
                    );

                    if current_i_bits_left <= fitting {
                        current_i += 1;
                        current_i_bits_left = 64u32;
                    } else {
                        current_i_bits_left -= fitting;
                    }
                }
            }
            rest_bits = min(chimp.bit_count as u32, 64u32);
            fitting = CPUWriter64::get_fitting(rest_bits, current_i_bits_left);
            insert_index = CPUWriter64::get_insert_index(rest_bits, current_i_bits_left);
            remaining = CPUWriter64::get_remaining(rest_bits, current_i_bits_left);

            bits_to_add = extract_bits(chimp.lower_bits, rest_bits - fitting, fitting);
            self.out_vec[current_i as usize] = insert_bits(
                self.out_vec[current_i as usize],
                bits_to_add,
                insert_index,
                fitting,
            );

            if current_i_bits_left <= fitting {
                current_i += 1u32;
                current_i_bits_left = 64u32;
            } else {
                current_i_bits_left -= fitting;
            }
            if remaining > 0 {
                fitting = CPUWriter64::get_fitting(remaining, current_i_bits_left);
                insert_index = CPUWriter64::get_insert_index(remaining, current_i_bits_left);
                bits_to_add = extract_bits(chimp.lower_bits, 0u32, fitting);
                self.out_vec[current_i as usize] = insert_bits(
                    self.out_vec[current_i as usize],
                    bits_to_add,
                    insert_index,
                    fitting,
                );
                if current_i_bits_left <= fitting {
                    current_i += 1u32;
                    current_i_bits_left = 64u32;
                } else {
                    current_i_bits_left -= fitting;
                }
            }
        }
        current_i
    }
}
