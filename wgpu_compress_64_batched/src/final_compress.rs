use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::{ChimpOutput64, S};
use compress_utils::{execute_compute_shader, step, wgpu_utils, BufferWrapper, WgpuGroupId};
use log::info;
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait FinalCompress: MaxGroupGnostic {
    async fn final_compress(
        &self,
        input: &mut Vec<f64>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> anyhow::Result<Vec<ChimpOutput64>>;
}

pub struct FinalCompressImpl64 {
    context: Arc<Context>,
    // debug: bool,
}

impl FinalCompressImpl64 {
    pub fn new(context: Arc<Context>, _debug: bool) -> Self {
        Self {
            context,
            // debug
        }
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref()
    }
}

impl MaxGroupGnostic for FinalCompressImpl64 {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompress for FinalCompressImpl64 {
    async fn final_compress(
        &self,
        input: &mut Vec<f64>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> anyhow::Result<Vec<ChimpOutput64>> {
        let utils_64 = include_str!("shaders/64_utils.wgsl");
        let temp = include_str!("shaders/chimp_compress.wgsl")
            .replace("//#include(64_utils)", utils_64)
            .to_string();
        let size_of_s = size_of::<S>();
        let size_of_output = size_of::<ChimpOutput64>();
        let input_length = input.len();
        info!("The length of the input vec: {}", input_length);

        let s_buffer_size = (size_of_s * s_values.len()) as BufferAddress;
        info!("The S buffer size in bytes: {}", &s_buffer_size);

        let output_buffer_size = (size_of_output * s_values.len()) as BufferAddress;
        info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = self.get_max_number_of_groups(input.len());
        info!("The wgpu workgroup size: {}", &workgroup_count);
        let output_staging_buffer = BufferWrapper::stage_with_size(
            self.context().device(),
            output_buffer_size,
            Some("Staging S Buffer"),
        );
        let output_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size,
            WgpuGroupId::new(0, 2),
            Some("Storage Output Buffer"),
        );
        let s_storage_buffer = BufferWrapper::storage_with_content(
            self.context().device(),
            bytemuck::cast_slice(s_values.as_slice()),
            WgpuGroupId::new(0, 0),
            Some("Storage S Buffer"),
        );
        input.push(0f64);
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.context().device(),
            bytemuck::cast_slice(input.as_slice()),
            WgpuGroupId::new(0, 1),
            Some("Storage Input Buffer"),
        );
        let chunks_buffer = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&ChimpBufferInfo::get().chunks()),
            WgpuGroupId::new(0, 3),
            Some("Chunks Buffer"),
        );
        execute_compute_shader!(
            self.context(),
            &temp,
            vec![
                &s_storage_buffer,
                &input_storage_buffer,
                &output_storage_buffer,
                &output_staging_buffer,
                &chunks_buffer,
            ],
            workgroup_count,
            Some("compress pass")
        );

        let output = wgpu_utils::get_s_output::<ChimpOutput64>(
            self.context(),
            output_storage_buffer.buffer(),
            output_buffer_size,
            output_staging_buffer.buffer(),
        )
        .await?;

        let length_without_padding = output.len() - padding - 1;

        let mut final_output = Vec::<ChimpOutput64>::with_capacity(length_without_padding);
        final_output.extend(output[0..length_without_padding].to_vec());
        for i in 0..workgroup_count {
            let index = i * ChimpBufferInfo::get().buffer_size();
            let mut c = ChimpOutput64::default();
            c.set_lower_bits(bytemuck::cast(input[index]));
            c.set_bit_count(64);

            final_output[index] = c;
        }
        step!(Step::Compress, {
            final_output
                .iter()
                .enumerate()
                .map(|it| format!("{}:{}\n", it.0, it.1))
                .into_iter()
        });
        Ok(final_output)
    }
}
