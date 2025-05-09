use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{get_buffer_size, trace_steps, Step};
use compress_utils::types::ChimpOutput;
use compress_utils::{wgpu_utils, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use log::info;
use std::cmp::{max, min};
use std::ops::Div;
use std::sync::Arc;
use std::{fs, vec};
use wgpu_types::BufferAddress;

#[async_trait]
pub trait Finalize {
    async fn finalize(
        &self,
        chimp_output: &mut Vec<ChimpOutput>,
        padding: usize,
    ) -> Result<Vec<u8>>;
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
        chimp_input: &mut Vec<ChimpOutput>,
        padding: usize,
    ) -> Result<Vec<u8>> {
        let temp = include_str!("shaders/chimp_finalize_compress.wgsl").to_string();
        let final_compress_module = wgpu_utils::create_shader_module(self.device(), &temp)?;
        // let size_of_chimp = size_of::<ChimpOutput>();
        let size_of_out = size_of::<u32>();

        let chimp_input_length = chimp_input.len() - padding;
        let input_length = chimp_input_length;
        info!("The length of the input vec: {}", input_length);

        let output_buffer_size = (size_of_out * chimp_input_length) as BufferAddress;
        info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = chimp_input.len().div(get_buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let out_stage_buffer = BufferWrapper::stage_with_size(
            self.device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let out_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            output_buffer_size,
            WgpuGroupId::new(0, 0),
            Some("Staging Output Buffer"),
        );
        let in_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(chimp_input.as_slice()),
            WgpuGroupId::new(0, 1),
            Some("Staging Output Buffer"),
        );
        let size_uniform = BufferWrapper::uniform_with_content(
            self.device(),
            bytemuck::cast_slice(get_buffer_size().to_ne_bytes().as_slice()),
            WgpuGroupId::new(0, 2),
            Some("Size Uniform Buffer"),
        );

        let useful_byte_count_storage = BufferWrapper::storage_with_size(
            self.device(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            WgpuGroupId::new(0, 3),
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
                &useful_byte_count_staging,
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
            let start_index = i * get_buffer_size();
            let byte_count = min(*useful_byte_count as usize, chimp_input_length - 1);
            let temp_vec = output[start_index..=byte_count]
                .iter()
                .flat_map(|it| it.to_be_bytes())
                .collect_vec();

            let batch_size =
                if i == workgroup_count - 1 && chimp_input_length % get_buffer_size() != 0 {
                    ((chimp_input_length % get_buffer_size()) - 1) as u8
                } else {
                    (get_buffer_size() - 1) as u8
                };
            final_vec.extend(batch_size.to_be_bytes());
            final_vec.extend((temp_vec.len() as u32).to_be_bytes().iter());
            final_vec.extend(temp_vec);
        }
        if trace_steps().contains(&Step::Finalize) {
            let trace_path = Step::Finalize.get_trace_file();
            let mut trace_output = String::new();

            final_vec
                .iter()
                .for_each(|it| trace_output.push_str(&format!("{:08b}", it)));

            fs::write(&trace_path, trace_output)?;
        }
        Ok(final_vec)
    }
}
