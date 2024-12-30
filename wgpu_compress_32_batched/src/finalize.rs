use anyhow::Result;
use async_trait::async_trait;
use bytemuck::Contiguous;
use compress_utils::context::Context;
use compress_utils::general_utils::get_buffer_size;
use compress_utils::types::ChimpOutput;
use compress_utils::{wgpu_utils, BufferWrapper};
use itertools::Itertools;
use log::info;
use std::cmp::{max, min};
use std::ops::Div;
use wgpu_types::BufferAddress;

#[async_trait]
pub(crate) trait Finalize {
    async fn finalize(&self, chimp_output: &mut Vec<ChimpOutput>) -> Result<Vec<u8>>;
}

pub(crate) struct Finalizer<'a> {
    context: &'a Context,
}

impl<'a> Finalizer<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &'a Context {
        self.context
    }

    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }
}

#[async_trait]
impl<'a> Finalize for Finalizer<'a> {
    async fn finalize(&self, chimp_output: &mut Vec<ChimpOutput>) -> Result<Vec<u8>> {
        let temp = include_str!("shaders/chimp_finalize_compress.wgsl").to_string();
        let final_compress_module = wgpu_utils::create_shader_module(self.device(), &temp)?;
        // let size_of_chimp = size_of::<ChimpOutput>();
        let size_of_out = size_of::<u32>();

        let buffer_size = chimp_output.len(); //get_buffer_size();

        let input_length = chimp_output.len();
        info!("The length of the input vec: {}", input_length);

        let output_buffer_size = (size_of_out * chimp_output.len()) as BufferAddress;
        info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = chimp_output.len().div(get_buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let out_stage_buffer = BufferWrapper::stage_with_size(
            self.device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let out_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let in_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(chimp_output.as_slice()),
            Some("Staging Output Buffer"),
        );
        let size_uniform = BufferWrapper::uniform_with_content(
            self.device(),
            bytemuck::cast_slice(buffer_size.to_ne_bytes().as_slice()),
            Some("Size Uniform Buffer"),
        );

        let useful_byte_count_storage = BufferWrapper::storage_with_size(
            self.device(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            Some("Useful Storage Buffer"),
        );
        let useful_byte_count_staging = BufferWrapper::stage_with_size(
            self.device(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            Some("Useful Staging Buffer"),
        );

        let binding_group_layout = wgpu_utils::assign_bind_groups(
            self.device(),
            vec![
                &out_stage_buffer,
                &out_storage_buffer,
                &in_storage_buffer,
                &size_uniform,
                &useful_byte_count_storage,
                &useful_byte_count_staging,
            ],
        );
        let improve_s_pipeline = wgpu_utils::create_compute_shader_pipeline(
            self.device(),
            &final_compress_module,
            &binding_group_layout,
            Some("Compress pipeline"),
        )?;
        let binding_group = wgpu_utils::create_bind_group(
            self.context(),
            &binding_group_layout,
            vec![
                &out_stage_buffer,
                &out_storage_buffer,
                &in_storage_buffer,
                &size_uniform,
                &useful_byte_count_storage,
            ],
        );
        let mut s_encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut s_pass = s_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compress_pass"),
                timestamp_writes: None,
            });
            s_pass.set_pipeline(&improve_s_pipeline);
            s_pass.set_bind_group(0, &binding_group, &[]);
            s_pass.dispatch_workgroups(max(workgroup_count, 1) as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));

        let output = wgpu_utils::get_s_output::<u32>(
            self.context(),
            out_storage_buffer.buffer(),
            output_buffer_size,
            out_stage_buffer.buffer(),
        )
        .await?;
        let indexes = wgpu_utils::get_s_output::<u32>(
            self.context(),
            useful_byte_count_storage.buffer(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            useful_byte_count_staging.buffer(),
        )
        .await?;
        let mut final_vec = Vec::<u8>::new();
        for (i, useful_byte_count) in indexes.iter().enumerate() {
            let start_index = i * buffer_size;
            let byte_count = *useful_byte_count as usize;
            // for num in &output[start_index..start_index + byte_count + 1] {
            //     println!("{}", num);
            // }
            final_vec.extend(
                output[start_index..start_index + byte_count + 1]
                    .iter()
                    .flat_map(|it| it.to_ne_bytes()),
            );
        }
        Ok(final_vec)
    }
}

#[derive(Debug, Default)]
pub struct CPUImpl {}
struct CPUImplHelper {
    size: u32,
    last_size: Vec<u32>,
    out: Vec<u32>,
    chimp_input: Vec<ChimpOutput>,
}

impl CPUImplHelper {
    fn write(&mut self, index: usize) -> u32 {
        let mut current_i = 1u32;
        let mut current_i_bits_left = 32u32;

        let mut insert_index = 0u32;

        let mut rest_bits = 0u32;
        let mut rest_fit = 0;

        self.out[0] = self.chimp_input[0].content_y();
        for i in 1..index + (self.size as usize) {
            let chimp: ChimpOutput = self.chimp_input[i];
            let overflow_bits = (chimp.bit_count() as i32) - 32;
            let current_str1 = Self::format_u32(self.out[current_i as usize]);
            let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

            if overflow_bits > 0 {
                let fitting = get_fitting(overflow_bits as u32, current_i_bits_left);
                let insert_index = get_insert_index(overflow_bits as u32, current_i_bits_left);
                let bits_to_add = extractBits(
                    chimp.content_x(),
                    (overflow_bits - fitting as i32) as u32,
                    fitting,
                );
                let current_str3 = Self::format_u32(bits_to_add);
                let current_str4 = Self::format_u32(chimp.content_y());
                self.out[current_i as usize] = insertBits(
                    self.out[current_i as usize],
                    bits_to_add,
                    insert_index,
                    fitting,
                );
                let current_str1 = Self::format_u32(self.out[current_i as usize]);
                let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

                if current_i_bits_left <= fitting {
                    current_i += 1;
                    current_i_bits_left = 32u32
                } else {
                    current_i_bits_left -= fitting;
                }
                let remaining = get_remaining(overflow_bits as u32, current_i_bits_left);
                if remaining > 0 {
                    let fitting = get_fitting(remaining, current_i_bits_left);
                    let insert_index = get_insert_index(remaining, current_i_bits_left);
                    let bits_to_add = extractBits(chimp.content_x(), 0, fitting);
                    let current_str3 = Self::format_u32(bits_to_add);
                    let current_str4 = Self::format_u32(chimp.content_y());
                    self.out[current_i as usize] = insertBits(
                        self.out[current_i as usize],
                        bits_to_add,
                        insert_index,
                        fitting,
                    );
                    let current_str1 = Self::format_u32(self.out[current_i as usize]);
                    let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

                    if current_i_bits_left <= fitting {
                        current_i += 1;
                        current_i_bits_left = 32u32
                    } else {
                        current_i_bits_left -= fitting;
                    }
                }
            }
            rest_bits = min(chimp.bit_count(), 32u32);
            let fitting = get_fitting(rest_bits, current_i_bits_left);
            assert!(rest_bits >= fitting);
            let insert_index = get_insert_index(rest_bits, current_i_bits_left);
            let remaining = get_remaining(rest_bits, current_i_bits_left);
            let bits_to_add = extractBits(chimp.content_y(), rest_bits - fitting, fitting);
            let current_str3 = Self::format_u32(bits_to_add);
            let current_str4 = Self::format_u32(chimp.content_y());

            self.out[current_i as usize] = insertBits(
                self.out[current_i as usize],
                bits_to_add,
                insert_index,
                fitting,
            );
            let current_str1 = Self::format_u32(self.out[current_i as usize]);
            let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

            if current_i_bits_left <= fitting {
                current_i += 1;
                current_i_bits_left = 32u32
            } else {
                current_i_bits_left -= fitting;
            }
            if remaining > 0 {
                let fitting = get_fitting(remaining, current_i_bits_left);
                let insert_index = get_insert_index(remaining, current_i_bits_left);
                let bits_to_add = extractBits(chimp.content_y(), 0, fitting);
                let current_str3 = Self::format_u32(bits_to_add);
                let current_str4 = Self::format_u32(chimp.content_y());
                self.out[current_i as usize] = insertBits(
                    self.out[current_i as usize],
                    bits_to_add,
                    insert_index,
                    fitting,
                );
                let current_str1 = Self::format_u32(self.out[current_i as usize]);
                let current_str2 = Self::format_u32(self.out[max(current_i - 1, 0) as usize]);

                if current_i_bits_left <= fitting {
                    current_i += 1;
                    current_i_bits_left = 32u32
                } else {
                    current_i_bits_left -= fitting;
                }
            }
        }
        current_i
    }

    fn format_u32(p0: u32) -> String {
        format!("{:032b}", p0)
    }
}
#[async_trait]
impl Finalize for CPUImpl {
    async fn finalize(&self, chimp_output: &mut Vec<ChimpOutput>) -> Result<Vec<u8>> {
        let out = vec![0; (1.5 * (chimp_output.len() as f32)).floor() as usize];
        let workgroup_count = chimp_output.len() / 256;
        let last_size = vec![0; workgroup_count];
        let mut helper = CPUImplHelper {
            chimp_input: chimp_output.to_owned(),
            size: 256,
            out,
            last_size,
        };

        for i in 0..workgroup_count {
            helper.last_size[i] = helper.write(i)
        }
        let mut final_output = Vec::new();

        for i in 0..workgroup_count {
            let final_iter =
                helper.out[(i * (helper.size as usize))..(helper.last_size[i] as usize)].iter();
            let final_byte_vec = final_iter.flat_map(|it| it.to_ne_bytes()).collect_vec();
            final_output.extend(final_byte_vec);
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

fn insertBits(input_bits: u32, new_bits: u32, start_index: u32, bit_count: u32) -> u32 {
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

fn extractBits(input_bits: u32, start_index: u32, bit_count: u32) -> u32 {
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
    use crate::finalize::{extractBits, insertBits};

    #[test]
    fn test1() {
        let u = 127u32;
        assert_eq!(extractBits(u, 3, 3), 7);
    }
    #[test]
    fn test3() {
        assert_eq!(7 / 3, 2);
    }

    #[test]
    fn test2() {
        let u = 113u32;
        let new_u = insertBits(u, 15, 1, 3);
        assert_eq!(new_u, 127);
    }
}
