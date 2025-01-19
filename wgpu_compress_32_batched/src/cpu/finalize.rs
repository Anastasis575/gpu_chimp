use crate::finalize::Finalize;
use async_trait::async_trait;
use bytemuck::Contiguous;
use compress_utils::general_utils::{get_buffer_size, trace_steps, Step};
use compress_utils::types::ChimpOutput;
use itertools::Itertools;
use std::cmp::{max, min};
use std::fs;

#[derive(Debug, Default)]
pub struct CPUImpl {}
struct CPUImplHelper {
    size: u32,
    last_size: Vec<u32>,
    out: Vec<u32>,
    chimp_input: Vec<ChimpOutput>,
}

impl CPUImplHelper {
    // #[allow(unused_variables)]

    /// Write to the output array starting from an offset set as dictated by a previous call,
    /// Starts
    fn write(&mut self, out_offset: usize, in_offset: usize) -> u32 {
        let mut current_i = (out_offset as u32) + 1u32;
        let mut current_i_bits_left = 32u32;

        self.out[out_offset] = self.chimp_input[in_offset].lower_bits();
        for i in 1 + in_offset..in_offset + (self.size as usize) {
            let chimp: ChimpOutput = self.chimp_input[i];
            let overflow_bits = (chimp.bit_count() as i32) - 32;
            // let current_str1 = Self::format_u32(self.out[current_i as usize]);
            // let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

            if overflow_bits > 0 {
                let fitting = get_fitting(overflow_bits as u32, current_i_bits_left);
                let insert_index = get_insert_index(overflow_bits as u32, current_i_bits_left);
                let remaining = get_remaining(overflow_bits as u32, current_i_bits_left);
                let bits_to_add = extract_bits(
                    chimp.upper_bits(),
                    (overflow_bits - fitting as i32) as u32,
                    fitting,
                );
                // let current_str3 = Self::format_u32(bits_to_add);
                // let current_str4 = Self::format_u32(chimp.lower_bits());
                self.out[current_i as usize] = insert_bits(
                    self.out[current_i as usize],
                    bits_to_add,
                    insert_index,
                    fitting,
                );
                // let current_str1 = Self::format_u32(self.out[current_i as usize]);
                // let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

                if current_i_bits_left <= fitting {
                    // println!(
                    //     "current_i:{} output:{:032b}",
                    //     current_i, self.out[current_i as usize]
                    // );
                    current_i += 1;
                    current_i_bits_left = 32u32
                } else {
                    current_i_bits_left -= fitting;
                }
                if remaining > 0 {
                    let fitting = get_fitting(remaining, current_i_bits_left);
                    let insert_index = get_insert_index(remaining, current_i_bits_left);
                    let bits_to_add = extract_bits(chimp.upper_bits(), 0, fitting);
                    // let current_str3 = Self::format_u32(bits_to_add);
                    // let current_str4 = Self::format_u32(chimp.lower_bits());
                    self.out[current_i as usize] = insert_bits(
                        self.out[current_i as usize],
                        bits_to_add,
                        insert_index,
                        fitting,
                    );
                    // let current_str1 = Self::format_u32(self.out[current_i as usize]);
                    // let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

                    if current_i_bits_left <= fitting {
                        // println!(
                        //     "current_i:{} output:{:032b}",
                        //     current_i, self.out[current_i as usize]
                        // );
                        current_i += 1;
                        current_i_bits_left = 32u32
                    } else {
                        current_i_bits_left -= fitting;
                    }
                }
            }
            let rest_bits = min(chimp.bit_count(), 32u32);
            let fitting = get_fitting(rest_bits, current_i_bits_left);
            assert!(rest_bits >= fitting);
            let insert_index = get_insert_index(rest_bits, current_i_bits_left);
            let remaining = get_remaining(rest_bits, current_i_bits_left);
            let bits_to_add = extract_bits(chimp.lower_bits(), rest_bits - fitting, fitting);
            // let current_str3 = Self::format_u32(bits_to_add);
            // let current_str4 = Self::format_u32(chimp.lower_bits());

            self.out[current_i as usize] = insert_bits(
                self.out[current_i as usize],
                bits_to_add,
                insert_index,
                fitting,
            );
            // let current_str1 = Self::format_u32(self.out[current_i as usize]);
            // let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

            if current_i_bits_left <= fitting {
                // println!(
                //     "current_i:{} output:{:032b}",
                //     current_i, self.out[current_i as usize]
                // );
                current_i += 1;
                current_i_bits_left = 32u32
            } else {
                current_i_bits_left -= fitting;
            }
            if remaining > 0 {
                let fitting = get_fitting(remaining, current_i_bits_left);
                let insert_index = get_insert_index(remaining, current_i_bits_left);
                let bits_to_add = extract_bits(chimp.lower_bits(), 0, fitting);
                // let current_str3 = Self::format_u32(bits_to_add);
                // let current_str4 = Self::format_u32(chimp.lower_bits());
                self.out[current_i as usize] = insert_bits(
                    self.out[current_i as usize],
                    bits_to_add,
                    insert_index,
                    fitting,
                );
                // let current_str1 = Self::format_u32(self.out[current_i as usize]);
                // let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

                if current_i_bits_left <= fitting {
                    // println!(
                    //     "current_i:{} output:{:032b}",
                    //     current_i, self.out[current_i as usize]
                    // );
                    current_i += 1;
                    current_i_bits_left = 32u32
                } else {
                    current_i_bits_left -= fitting;
                }
            }
        }
        // println!(
        //     "current_i:{} output:{:032b}",
        //     current_i, self.out[current_i as usize]
        // );
        current_i
    }

    fn format_u32(p0: u32) -> String {
        format!("{:032b}", p0)
    }
}
#[async_trait]
impl Finalize for CPUImpl {
    async fn finalize(&self, chimp_output: &mut Vec<ChimpOutput>) -> anyhow::Result<Vec<u8>> {
        //The number of iterations
        let workgroup_count = chimp_output.len() / get_buffer_size();

        // The output needs at worst  twice the number of 32-bit numbers to be coded along with one
        // space for the size of the workgroup in bytes
        // We come across this instance when each consecutive number in the series is too different from the other.
        let out = vec![0; workgroup_count + (2 * chimp_output.len())];

        // The index of the final usable u32 in the out vector
        let last_size = vec![0; workgroup_count];

        let mut helper = CPUImplHelper {
            chimp_input: chimp_output.to_owned(),
            size: get_buffer_size() as u32,
            out,
            last_size,
        };

        //for each workgroup write the bytes concec
        let mut current_index = 0u32;
        for i in 0..workgroup_count {
            current_index = helper.write(current_index as usize, i * get_buffer_size());
            helper.last_size[i] = current_index;
            current_index += 1;
        }
        let mut final_output = Vec::new();

        for i in 0..workgroup_count {
            let start_index = if i == 0 {
                0
            } else {
                helper.last_size[i - 1] + 1
            };
            let final_iter =
                helper.out[start_index as usize..=(helper.last_size[i] as usize)].to_vec();
            let final_byte_vec = final_iter
                .iter()
                .flat_map(|it| it.to_be_bytes())
                .collect_vec();
            final_output.extend((get_buffer_size() as u32).to_be_bytes());
            final_output.extend((final_byte_vec.len() as u32).to_be_bytes());
            final_output.extend(&final_byte_vec);
        }

        if trace_steps().contains(&Step::Finalize) {
            let trace_path = Step::Finalize.get_trace_file();
            let mut trace_output = String::new();

            final_output
                .iter()
                .enumerate()
                .for_each(|it| trace_output.push_str(&format!("{}:{:08b}\n", it.0, it.1)));

            fs::write(&trace_path, trace_output)?;
        }
        Ok(final_output)
    }
}

fn get_fitting(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
    min(bits_rest_to_write, writeable_output_remaining)
}

fn get_remaining(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
    max(
        bits_rest_to_write - get_fitting(bits_rest_to_write, writeable_output_remaining),
        0,
    )
}

fn get_insert_index(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
    max(
        writeable_output_remaining - get_fitting(bits_rest_to_write, writeable_output_remaining),
        0,
    )
}

fn insert_bits(input_bits: u32, new_bits: u32, start_index: u32, bit_count: u32) -> u32 {
    let mut output_bits = 0u32;

    let end_index = min(start_index + bit_count, 32);
    let copiable_values = end_index - start_index;

    let bits_to_copy = new_bits % 2u32.pow(copiable_values);

    if end_index < 32 {
        // let to_end = 32 - end_index;
        output_bits += input_bits >> end_index;
        output_bits <<= copiable_values;
    }
    output_bits += bits_to_copy;
    output_bits <<= start_index;
    if start_index != 0 {
        output_bits += input_bits % 2u32.pow(start_index);
    }
    output_bits
}

fn extract_bits(input_bits: u32, start_index: u32, bit_count: u32) -> u32 {
    let mut input_bits = input_bits;
    let end_index = min(start_index + bit_count, 32);
    let low_bound = u32::MAX_VALUE << start_index;
    let high_bound = u32::MAX_VALUE >> (32 - end_index);

    input_bits &= low_bound;
    input_bits &= high_bound;
    input_bits >> start_index
}

#[cfg(test)]
mod temp_test {
    use crate::cpu::finalize::{extract_bits, insert_bits};

    #[test]
    fn test1() {
        let u = 127u32;
        assert_eq!(extract_bits(u, 3, 3), 7);
    }
    #[test]
    fn test3() {
        assert_eq!(7 / 3, 2);
    }

    #[test]
    fn test2() {
        let u = 113u32;
        let new_u = insert_bits(u, 15, 1, 3);
        assert_eq!(new_u.to_ne_bytes(), 127u32.to_ne_bytes());
    }

    #[test]
    fn test22() {
        let u = 1757806588u32;
        let extracted = extract_bits(u, 0, 31);
        let new_u = insert_bits(0u32, extracted, 1, 31);
        assert_eq!(new_u, 3515613177u32);
    }
    #[test]
    fn testend() {
        assert_eq!(125u32.to_ne_bytes(), 125u32.to_be_bytes());
        assert_eq!(125u32.to_ne_bytes(), 125u32.to_le_bytes());
    }
}
