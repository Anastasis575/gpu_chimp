use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, CompressResult, Step};
use compress_utils::types::ChimpOutput64;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, step, wgpu_utils, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::cmp::max;
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use std::{fs, vec};
use wgpu_types::BufferAddress;

#[async_trait]
pub trait Finalize {
    async fn finalize(
        &self,
        buffers: &mut RunBuffers,
        padding: usize,
        skip_time: &mut u128,
    ) -> Result<CompressResult>;
}

#[derive(Debug)]
pub struct Finalizer64 {
    context: Arc<Context>,
}

impl Finalizer64 {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }
    pub fn context(&self) -> &Context {
        &self.context
    }
}

#[async_trait]
impl Finalize for Finalizer64 {
    async fn finalize(
        &self,
        buffers: &mut RunBuffers,
        padding: usize,
        skip_time: &mut u128,
    ) -> Result<CompressResult> {
        let util_64 = include_str!("shaders/64_utils.wgsl");
        let temp = include_str!("shaders/chimp_finalize_compress.wgsl")
            .replace("//#include(64_utils)", util_64)
            .to_string();
        let instant = Instant::now();
        let index_staging = BufferWrapper::stage_with_size(
            self.context().device(),
            buffers.index_buffer().size() as BufferAddress,
            None,
        );

        let indexes = wgpu_utils::get_from_gpu::<u32>(
            self.context(),
            buffers.index_buffer().buffer(),
            buffers.index_buffer().size() as BufferAddress,
            index_staging.buffer(),
        )
        .await?;

        let chimp_input_len = buffers.compressed_buffer().size() / size_of::<ChimpOutput64>();
        let chimp_input_length_no_padding = chimp_input_len - padding;
        let size = ChimpBufferInfo::get().buffer_size() as u32;
        let last_size =
            if (chimp_input_length_no_padding % ChimpBufferInfo::get().buffer_size() == 0) {
                ChimpBufferInfo::get().buffer_size()
            } else {
                chimp_input_length_no_padding % ChimpBufferInfo::get().buffer_size()
            } as u32
                - 2;

        let output_buffer_size =
            ((*indexes.last().unwrap() + 1) as usize * size_of::<u64>()) as BufferAddress;

        let workgroup_count = chimp_input_len.div(ChimpBufferInfo::get().buffer_size());

        let out_stage_buffer = BufferWrapper::stage_with_size(
            self.context().device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let out_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size,
            WgpuGroupId::new(0, 0),
            Some("Staging Output Buffer"),
        );
        {
            buffers
                .compressed_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 1));
        }
        {
            buffers
                .index_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 3));
        }
        let size_uniform = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&size),
            WgpuGroupId::new(0, 2),
            Some("Size Uniform Buffer"),
        );
        let last_size_uniform = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&last_size),
            WgpuGroupId::new(0, 4),
            Some("Last buffer size Uniform Buffer"),
        );
        *skip_time += instant.elapsed().as_millis();
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let last_pass = format!(
                "let last_pass={}u;",
                if i == iterations - 1 { 1 } else { 0 }
            );
            let util_64 = include_str!("shaders/64_utils.wgsl");
            let temp = include_str!("shaders/chimp_finalize_compress.wgsl")
                .replace("//#include(64_utils)", util_64)
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@last_pass", &last_pass)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &temp,
                vec![
                    &out_stage_buffer,
                    &out_storage_buffer,
                    buffers.compressed_buffer(),
                    &size_uniform,
                    buffers.index_buffer(),
                    &last_size_uniform,
                ],
                if i == iterations - 1 {
                    last_size
                } else {
                    self.context.get_max_workgroup_size()
                },
                Some("trim pass")
            );
        }
        let instant = Instant::now();
        let output = wgpu_utils::get_from_gpu::<u8>(
            self.context(),
            out_storage_buffer.buffer(),
            output_buffer_size,
            out_stage_buffer.buffer(),
        )
        .await?;
        *skip_time += instant.elapsed().as_millis();
        let mut final_vec = output;

        step!(&Step::Finalize, {
            final_vec
                .iter()
                .chunks(8)
                .into_iter()
                .map(|chunk| chunk.map(|it| format!("{:08b}", it)).join(" ") + "\n")
                .collect_vec()
                .into_iter()
        });
        Ok(CompressResult(final_vec, workgroup_count * 8, 0))
    }
}
