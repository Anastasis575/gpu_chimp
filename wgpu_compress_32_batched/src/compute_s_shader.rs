use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::S;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, wgpu_utils, BufferWrapper, WgpuGroupId};
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait ComputeS: MaxGroupGnostic {
    async fn compute_s(
        &self,
        values: &mut [f32],
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> Result<()>;
}

pub struct ComputeSImpl {
    context: Arc<Context>,
}

impl ComputeSImpl {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref()
    }
    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }
    #[allow(dead_code)]
    pub fn adapter(&self) -> &wgpu::Adapter {
        self.context.adapter()
    }
    #[allow(dead_code)]
    pub fn max_work_group_count(&self) -> usize {
        self.context.get_max_workgroup_size()
    }
}

impl MaxGroupGnostic for ComputeSImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl ComputeS for ComputeSImpl {
    async fn compute_s(
        &self,
        values: &mut [f32],
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> Result<()> {
        // Create a shader module and pipeline

        //Calculating buffer sizes and workgroup counts
        let workgroup_count = self.get_max_number_of_groups(values.len());
        //info!("The wgpu workgroup size: {}", &workgroup_count);

        let size_of_s = size_of::<S>();
        let bytes = values.len() + 1;
        //info!("The size of the input values vec: {}", bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        //info!("The S buffer size in bytes: {}", s_buffer_size);

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);

        let instant = Instant::now();
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(padded_values.as_slice()),
            WgpuGroupId::new(0, 1),
            Some("Storage Input Buffer"),
        );
        let s_staging_buffer =
            BufferWrapper::stage_with_size(self.device(), s_buffer_size, Some("Staging S Buffer"));
        let s_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            s_buffer_size,
            WgpuGroupId::new(0, 0),
            Some("Storage S Buffer"),
        );
        let chunks_buffer = BufferWrapper::uniform_with_content(
            self.device(),
            bytemuck::bytes_of(&ChimpBufferInfo::get().chunks()),
            WgpuGroupId::new(0, 2),
            Some("Chunks Buffer"),
        );
        *skip_time += instant.elapsed().as_millis();
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let temp = include_str!("shaders/compute_s.wgsl")
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
                    &s_storage_buffer,
                    &input_storage_buffer,
                    &s_staging_buffer,
                    &chunks_buffer
                ],
                if i == iterations - 1 {
                    last_size
                } else {
                    self.context.get_max_workgroup_size()
                },
                Some("compute s layout")
            );
        }
        buffers.set_input_buffer(input_storage_buffer);
        buffers.set_s_buffer(s_storage_buffer);
        buffers.set_chunks(chunks_buffer);

        //info!("Output result size: {}", output.len());
        if trace_steps().contains(&Step::ComputeS) {
            let output = wgpu_utils::get_from_gpu::<S>(
                self.context(),
                buffers.s_buffer().buffer(),
                s_buffer_size,
                s_staging_buffer.buffer(),
            )
            .await?;
            let trace_path = Step::ComputeS.get_trace_file();
            let mut trace_output = String::new();

            output.iter().for_each(|it| {
                trace_output.push_str(it.to_string().as_str());
                trace_output.push_str("\n");
            });

            fs::write(&trace_path, trace_output)?;
        }
        Ok(())
    }
}
