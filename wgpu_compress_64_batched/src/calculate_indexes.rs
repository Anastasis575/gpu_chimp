use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::Step;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo};
use compress_utils::types::ChimpOutput64;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, step, wgpu_utils, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait CalculateIndexes64 {
    async fn calculate_indexes(&self, input: &mut RunBuffers, size: u32) -> Result<()>;
}

pub struct GPUCalculateIndexes64 {
    context: Arc<Context>,
}

impl GPUCalculateIndexes64 {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
}

#[async_trait]
impl CalculateIndexes64 for GPUCalculateIndexes64 {
    async fn calculate_indexes(&self, buffers: &mut RunBuffers, size: u32) -> Result<()> {
        let util_64 = include_str!("shaders/64_utils.wgsl");
        let temp = include_str!("shaders/calculate_final_sizes.wgsl")
            .replace("//#include(64_utils)", util_64)
            .to_string();
        let input_len = buffers.compressed_buffer().size() / size_of::<ChimpOutput64>();
        let workgroup_count = input_len.div(size as usize);
        let output_buffer_size = (workgroup_count + 1) * size_of::<u32>();

        let mut out_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size as BufferAddress,
            WgpuGroupId::new(0, 0),
            Some("Staging Output Buffer"),
        );

        {
            buffers
                .compressed_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 1));
        }

        let size_uniform = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&size),
            WgpuGroupId::new(0, 2),
            Some("Size Uniform Buffer"),
        );
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let util_64 = include_str!("shaders/64_utils.wgsl");
            let temp = include_str!("shaders/calculate_final_sizes.wgsl")
                .replace("//#include(64_utils)", util_64)
                .replace(
                    "@@workgroup_size",
                    &ChimpBufferInfo::get().buffer_size().to_string(),
                )
                .replace("//@workgroup_offset", &offset_decl)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &temp,
                vec![
                    &out_storage_buffer,
                    buffers.compressed_buffer(),
                    &size_uniform
                ],
                if i == iterations - 1 {
                    last_size
                } else {
                    self.context.get_max_workgroup_size()
                },
                Some("calculate indexes pass")
            );
        }
        let size_uniform = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&workgroup_count),
            WgpuGroupId::new(0, 1),
            Some("Size Uniform Buffer"),
        );
        execute_compute_shader!(
            self.context(),
            r#"
             @group(0)
            @binding(0)
            var<storage,read_write> last_byte_index: array<u32>;
            @group(0)
            @binding(1)
            var<uniform> size: u32;
            @compute
            @workgroup_size(1)
            fn main(@builtin(num_workgroups) workgroup_id: vec3<u32>) {
                 for (var i=1u;i<size;i++){
                last_byte_index[i+1u]+=last_byte_index[i];
                }
                last_byte_index[0]=0u;
            }
            "#,
            vec![
                out_storage_buffer.with_binding(WgpuGroupId::new(0, 0)),
                &size_uniform
            ],
            1,
            Some("Add max")
        );

        buffers.set_index_buffer(out_storage_buffer);
        let mut output;
        step!(Step::CalculateIndexes, {
            let out_stage_buffer = BufferWrapper::stage_with_size(
                self.context().device(),
                buffers.index_buffer().size() as BufferAddress,
                None,
            );
            output = wgpu_utils::get_from_gpu::<u32>(
                self.context(),
                buffers.index_buffer().buffer(),
                buffers.index_buffer().size() as BufferAddress,
                out_stage_buffer.buffer(),
            )
            .await?;
            output.iter().map(|it| format!("{it}")).into_iter()
        });
        Ok(())
    }
}
