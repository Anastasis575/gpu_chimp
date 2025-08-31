use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::ChimpBufferInfo;
use compress_utils::types::ChimpOutput;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, wgpu_utils, BufferWrapper, WgpuGroupId};
use std::cmp::max;
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait CalculateIndexes {
    async fn calculate_indexes(
        &self,
        buffers: &mut RunBuffers,
        size: u32,
        skip_time: &mut u128,
    ) -> Result<()>;
}

pub struct GPUCalculateIndexes {
    context: Arc<Context>,
}

impl GPUCalculateIndexes {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
}

#[async_trait]
impl CalculateIndexes for GPUCalculateIndexes {
    async fn calculate_indexes(
        &self,
        buffers: &mut RunBuffers,
        size: u32,
        skip_time: &mut u128,
    ) -> Result<()> {
        let temp = include_str!("shaders/calculate_final_sizes.wgsl").to_string();
        let workgroup_count =
            (buffers.compressed_buffer().size() / size_of::<ChimpOutput>()).div(size as usize);
        let output_buffer_size = (workgroup_count + 1) * size_of::<u32>();
        // let out_stage_buffer = BufferWrapper::stage_with_size(
        //     self.context().device(),
        //     output_buffer_size as BufferAddress,
        //     Some("Staging Output Buffer"),
        // );
        let instant = Instant::now();
        let out_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size as BufferAddress,
            WgpuGroupId::new(0, 0),
            Some("Storage Output Buffer"),
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
        *skip_time += instant.elapsed().as_millis();
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let temp = include_str!("shaders/calculate_final_sizes.wgsl")
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

        // let mut output = wgpu_utils::get_s_output::<u32>(
        //     self.context(),
        //     out_storage_buffer.buffer(),
        //     output_buffer_size as BufferAddress,
        //     out_stage_buffer.buffer(),
        // )
        // .await?;

        // let out_stage_buffer = BufferWrapper::stage_with_size(
        //     self.context().device(),
        //     output_buffer_size as BufferAddress,
        //     Some("Staging Output Buffer"),
        // );
        // let out_storage_buffer = BufferWrapper::storage_with_content(
        //     self.context().device(),
        //     bytemuck::cast_slice(&output),
        //     WgpuGroupId::new(0, 0),
        //     Some("Staging Output Buffer"),
        // );
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
                last_byte_index[i+1]+=last_byte_index[i];
                }
                last_byte_index[0]=0u;
            }
            "#,
            vec![&out_storage_buffer, &size_uniform],
            1,
            Some("add sizes pass")
        );
        buffers.set_index_buffer(out_storage_buffer);
        // let mut output = wgpu_utils::get_s_output::<u32>(
        //     self.context(),
        //     out_storage_buffer.buffer(),
        //     output_buffer_size as BufferAddress,
        //     out_stage_buffer.buffer(),
        // )
        // .await?;
        // step!(Step::CalculateIndexes, {
        //     output.iter().map(|it| format!("{it}")).into_iter()
        // });
        Ok(())
    }
}
