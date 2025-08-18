use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::{ChimpOutput, S};
use compress_utils::{execute_compute_shader, wgpu_utils, BufferWrapper, WgpuGroupId};
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
        input: &mut Vec<f32>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> anyhow::Result<Vec<ChimpOutput>>;
}

pub struct FinalCompressImpl {
    context: Arc<Context>,
    // debug: bool,
}

impl FinalCompressImpl {
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

impl MaxGroupGnostic for FinalCompressImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompress for FinalCompressImpl {
    async fn final_compress(
        &self,
        input: &mut Vec<f32>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> anyhow::Result<Vec<ChimpOutput>> {
        let temp = include_str!("shaders/chimp_compress.wgsl").to_string();
        let size_of_s = size_of::<S>();
        let size_of_output = size_of::<ChimpOutput>();
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
        input.push(0f32);
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

        let output = wgpu_utils::get_s_output::<ChimpOutput>(
            self.context(),
            output_storage_buffer.buffer(),
            output_buffer_size,
            output_staging_buffer.buffer(),
        )
        .await?;

        let length_without_padding = output.len() - padding - 1;

        let mut final_output = Vec::<ChimpOutput>::new();
        final_output.extend(output[0..length_without_padding].to_vec());
        for i in 0..workgroup_count {
            let index = i * ChimpBufferInfo::get().buffer_size();
            let mut c = ChimpOutput::default();
            c.set_lower_bits(bytemuck::cast(input[index]));
            c.set_bit_count(32);

            final_output[index] = c;
        }
        if trace_steps().contains(&Step::Compress) {
            let trace_path = Step::Compress.get_trace_file();
            let mut trace_output = String::new();

            final_output.iter().enumerate().for_each(|it| {
                trace_output.push_str(&format!("{}:{}\n", it.0, it.1));
            });

            fs::write(&trace_path, trace_output)?;
        }
        Ok(final_output)
    }
}
