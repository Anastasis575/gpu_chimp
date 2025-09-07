use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, wgpu_utils, BufferWrapper, WgpuGroupId};
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait PreviousIndexes: MaxGroupGnostic {
    async fn calculate_previous_indexes(
        &self,
        values: &mut [f32],
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> Result<()>;
}

pub struct PreviousIndexesNImpl {
    context: Arc<Context>,
    n: usize,
}

impl PreviousIndexesNImpl {
    pub fn new(context: Arc<Context>, n: usize) -> Self {
        Self { context, n }
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

impl MaxGroupGnostic for PreviousIndexesNImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl PreviousIndexes for PreviousIndexesNImpl {
    async fn calculate_previous_indexes(
        &self,
        values: &mut [f32],
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> Result<()> {
        // Create a shader module and pipeline

        //Calculating buffer sizes and workgroup counts
        let workgroup_count = self.get_max_number_of_groups(values.len());
        //info!("The wgpu workgroup size: {}", &workgroup_count);

        let bytes = values.len() + 1;

        let previous_index_size = (size_of::<u32>() * bytes) as BufferAddress;

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);

        let instant = Instant::now();
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(padded_values.as_slice()),
            (0, 1),
            Some("Storage Input Buffer"),
        );

        let size_uniform = BufferWrapper::uniform_with_content(
            self.context.device(),
            bytemuck::bytes_of(&ChimpBufferInfo::get().buffer_size()),
            (0, 2),
            None,
        );
        let previous_index_buffer = BufferWrapper::storage_with_size(
            self.device(),
            previous_index_size,
            WgpuGroupId::new(0, 3),
            Some("Storage S Buffer"),
        );
        *skip_time += instant.elapsed().as_millis();

        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();

        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );

            let n = format!("let n={}u;", self.n);

            let full_size = format!("let full_size={}u;", ChimpBufferInfo::get().buffer_size());

            let index_size = format!("const indices_size={}u;", 2u32.pow(self.n.ilog2() + 1));

            let temp = include_str!("shaders/calculate_previous_efficient_index.wgsl")
                .replace(
                    "@@workgroup_size",
                    &ChimpBufferInfo::get().buffer_size().to_string(),
                )
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@n", &n)
                .replace("//@indices_size", &index_size)
                .replace("//@full_size", &full_size)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &temp,
                vec![&input_storage_buffer, &previous_index_buffer, &size_uniform],
                if i == iterations - 1 {
                    last_size
                } else {
                    self.context.get_max_workgroup_size()
                },
                Some("compute s layout")
            );
        }

        buffers.set_input_buffer(input_storage_buffer);
        buffers.set_previous_index_buffer(previous_index_buffer);

        //info!("Output result size: {}", output.len());
        if trace_steps().contains(&Step::PreviousIndexes) {
            let previous_index_staging = BufferWrapper::stage_with_size(
                self.context().device(),
                previous_index_size,
                Some("previous index staging buffer"),
            );

            let previous_index = wgpu_utils::get_from_gpu::<u32>(
                self.context(),
                buffers.previous_index_buffer().buffer(),
                buffers.previous_index_buffer().size() as BufferAddress,
                previous_index_staging.buffer(),
            )
            .await?;
            let trace_path = Step::PreviousIndexes.get_trace_file();
            let mut trace_output = String::new();

            previous_index.iter().for_each(|it| {
                let temp = format!("{}\n", it);
                trace_output.push_str(&temp);
            });

            fs::write(&trace_path, trace_output)?;
        }
        Ok(())
    }
}
