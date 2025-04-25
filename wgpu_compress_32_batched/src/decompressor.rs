use crate::ChimpCompressorBatched;
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::{get_buffer_size, trace_steps, MaxGroupGnostic, Step};
use compress_utils::types::S;
use compress_utils::{time_it, wgpu_utils, BufferWrapper, WgpuGroupId};
use log::info;
use std::cmp::max;
use std::fs;
use wgpu::{Device, Queue};
use wgpu_types::BufferAddress;
#[async_trait]
impl Decompressor for ChimpCompressorBatched {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>, DecompressionError> {
        let decompressor = BatchedGPUDecompressor::new(self.context());
        decompressor.decompress(vec).await
    }
}
#[async_trait]
impl<'a> Decompressor for BatchedGPUDecompressor<'a> {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>, DecompressionError> {
        let mut current_index = 0usize;
        let mut output = Vec::new();
        let mut total_millis = 0;
        time_it!(
            {
                while current_index < vec.len() {
                    let buffer_size = u8::from_be_bytes(
                        vec[current_index..current_index + size_of::<u8>()]
                            .try_into()
                            .unwrap(),
                    ) as usize
                        + 1;
                    current_index += size_of::<u8>();

                    let size = u32::from_be_bytes(
                        vec[current_index..current_index + size_of::<u32>()]
                            .try_into()
                            .unwrap(),
                    );
                    current_index += size_of::<u32>();
                    let vec_view = vec[current_index..current_index + (size as usize)].to_vec();
                    let bit_vec = BitVec::from_bytes(&vec_view);
                    let block_values = self.decompress_block(&vec_view, buffer_size).await?;

                    output.extend(block_values[0..buffer_size].iter());
                    current_index += size as usize;
                }
            },
            total_millis,
            "decompression"
        );
        Ok(output)
    }
}

pub struct BatchedGPUDecompressor<'a> {
    context: &'a Context,
}
impl MaxGroupGnostic for BatchedGPUDecompressor<'_> {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        get_buffer_size()
    }
}

impl<'a> BatchedGPUDecompressor<'a> {
    pub(crate) async fn decompress_block(
        &self,
        values: &[u8],
        buffer_size: usize,
    ) -> Result<Vec<f32>, DecompressionError> {
        let result = Vec::new();
        let temp = include_str!("shaders/decompress.wgsl");

        let decompress_module = wgpu_utils::create_shader_module(self.device(), &temp, WGSL)?;
        let workgroup_count = self.get_max_number_of_groups(values.len());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let size_of_s = size_of::<S>();
        let bytes = values.len() + 1;
        info!("The size of the input values vec: {}", bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        info!("The S buffer size in bytes: {}", s_buffer_size);

        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(values),
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

        let binding_group_layout = wgpu_utils::assign_bind_groups(
            self.device(),
            vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer],
        );

        let compute_s_pipeline = wgpu_utils::create_compute_shader_pipeline(
            self.device(),
            &decompress_module,
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
        if trace_steps().contains(&Step::Decompress) {
            let trace_path = Step::Decompress.get_trace_file();
            let mut trace_output = String::new();

            output
                .iter()
                .for_each(|it| trace_output.push_str(it.to_string().as_str()));

            fs::write(&trace_path, trace_output)
                .map_err(|it| DecompressionError::FromBaseAnyhowError(anyhow::anyhow!(it)))?;
        }
        Ok(result)
    }

    pub fn new(context: &'a compress_utils::context::Context) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn device(&self) -> &Device {
        &self.context.device()
    }
    pub fn queue(&self) -> &Queue {
        &self.context.queue()
    }
}
