use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::get_buffer_size;
use compress_utils::types::S;
use compress_utils::{wgpu_utils, BufferWrapper};
use log::info;
use std::cmp::max;
use std::ops::Div;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait ComputeS {
    async fn compute_s(&self, values: &mut [f32]) -> Result<Vec<S>>;
}

pub struct ComputeSImpl<'a> {
    context: &'a Context,
}

impl<'a> ComputeSImpl<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }
    pub fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }
}

#[async_trait]
impl<'a> ComputeS for ComputeSImpl<'a> {
    async fn compute_s(&self, values: &mut [f32]) -> Result<Vec<S>> {
        // Create shader module and pipeline
        let workgroup_size = format!("@workgroup_size({})", get_buffer_size());
        let temp = include_str!("shaders/compute_s.wgsl")
            .replace("#@workgroup_size(1)#", &workgroup_size)
            .to_string();
        let compute_s_shader_module = wgpu_utils::create_shader_module(self.device(), &temp)?;

        //Calculating buffer sizes and workgroup counts

        let size_of_s = size_of::<S>();
        let bytes = values.len() + 1;
        info!("The size of the input values vec: {}", bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        info!("The S buffer size in bytes: {}", s_buffer_size);

        let workgroup_count = values.len().div(get_buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(padded_values.as_slice()),
            Some("Storage Input Buffer"),
        );
        let s_staging_buffer =
            BufferWrapper::stage_with_size(self.device(), s_buffer_size, Some("Staging S Buffer"));
        let s_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            s_buffer_size,
            Some("Storage S Buffer"),
        );

        let binding_group_layout = wgpu_utils::assign_bind_groups(
            self.device(),
            vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer],
        );

        let compute_s_pipeline = wgpu_utils::create_compute_shader_pipeline(
            self.device(),
            &compute_s_shader_module,
            &binding_group_layout,
            Some("Compute s pipeline"),
        )?;
        let binding_group = wgpu_utils::create_bind_group(
            self.context(),
            &binding_group_layout,
            vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer],
        );

        let mut s_encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut s_pass = s_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("s_pass"),
                timestamp_writes: None,
            });
            s_pass.set_pipeline(&compute_s_pipeline);
            s_pass.set_bind_group(0, &binding_group, &[]);
            s_pass.dispatch_workgroups(max(workgroup_count, 1) as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));

        let output = wgpu_utils::get_s_output::<S>(
            self.context(),
            s_storage_buffer.buffer(),
            s_buffer_size,
            s_staging_buffer.buffer(),
        )
        .await?;
        info!("Output result size: {}", output.len());
        Ok(output)
    }
}
