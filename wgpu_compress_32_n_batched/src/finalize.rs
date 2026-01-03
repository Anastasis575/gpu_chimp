use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, CompressResult, Step};
use compress_utils::types::ChimpOutput;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{
    execute_compute_shader, general_utils, step, wgpu_utils, BufferWrapper, WgpuGroupId,
};
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
    ) -> Result<general_utils::CompressResult>;
}

#[derive(Debug)]
pub struct Finalizer {
    context: Arc<Context>,
}

impl Finalizer {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref()
    }

    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }
}

#[async_trait]
impl Finalize for Finalizer {
    async fn finalize(
        &self,
        buffers: &mut RunBuffers,
        padding: usize,
        skip_time: &mut u128,
    ) -> Result<CompressResult> {
        let size_of_out = size_of::<u32>();

        let index_len = buffers.index_buffer().size() / size_of::<u32>();
        let chimp_input_len = buffers.compressed_buffer().size() / size_of::<ChimpOutput>() - 1;

        //Count the metadata size
        let metadata_size_in_bytes = (index_len - 1) * 8;

        let chimp_input_length_no_padding = chimp_input_len - padding;

        let size = ChimpBufferInfo::get().buffer_size();

        let last_size = if chimp_input_length_no_padding % ChimpBufferInfo::get().buffer_size() != 0
        {
            chimp_input_length_no_padding % ChimpBufferInfo::get().buffer_size()
        } else {
            size
        } as u32;

        //info!("The length of the input vec: {}", chimp_input_length);
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

        let output_buffer_size =
            (size_of_out * (*indexes.last().unwrap() as usize)) as BufferAddress;
        //info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = chimp_input_len.div(ChimpBufferInfo::get().buffer_size());
        //info!("The wgpu workgroup size: {}", &workgroup_count);

        let instant = Instant::now();
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
        // let in_storage_buffer = buffers.compressed_buffer();
        //     BufferWrapper::storage_with_content(
        //     self.device(),
        //     bytemuck::cast_slice(chimp_input.as_slice()),
        //     WgpuGroupId::new(0, 1),
        //     Some("Staging Output Buffer"),
        // );
        let size_uniform = BufferWrapper::uniform_with_content(
            self.device(),
            bytemuck::bytes_of(&(ChimpBufferInfo::get().buffer_size() as u32)),
            WgpuGroupId::new(0, 2),
            Some("Size Uniform Buffer"),
        );
        // let useful_byte_count_storage =
        {
            buffers
                .index_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 3));
        }
        //     BufferWrapper::storage_with_content(
        //     self.device(),
        //     bytemuck::cast_slice(&indexes),
        //     WgpuGroupId::new(0, 3),
        //     Some("Useful Storage Buffer"),
        // );
        let last_size_uniform = BufferWrapper::uniform_with_content(
            self.device(),
            bytemuck::bytes_of(&last_size),
            WgpuGroupId::new(0, 4),
            Some("Useful Staging Buffer"),
        );
        *skip_time += instant.elapsed().as_millis();

        let iterations = workgroup_count / (256 * self.context.get_max_workgroup_size()) + 1;
        let last_size = workgroup_count % (256 * self.context.get_max_workgroup_size());
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let last_pass = format!(
                "let last_pass={}u;",
                if i == iterations - 1 { 1 } else { 0 }
            );
            let total_threads = format!("let total_threads={}u;", workgroup_count);
            let last_index = format!("let last_index={}u;", (*indexes).last().unwrap());
            let temp = include_str!("shaders/chimp_finalize_compress.wgsl")
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@last_pass", &last_pass)
                .replace("//@total_threads", &total_threads)
                .replace("//@last_index", &last_index)
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
                    last_size.div_ceil(256)
                } else {
                    self.context.get_max_workgroup_size().div_ceil(256)
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

        let final_vec = output;
        step!(&Step::Finalize, {
            final_vec
                .iter()
                .chunks(4)
                .into_iter()
                .map(|chunk| chunk.map(|it| format!("{:08b}", it)).join(" ") + "\n")
                .collect_vec()
                .into_iter()
        });
        Ok(CompressResult(final_vec, metadata_size_in_bytes, 0))
    }
}
