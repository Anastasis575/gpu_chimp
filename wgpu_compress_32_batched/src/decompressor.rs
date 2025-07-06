use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::{get_buffer_size, trace_steps, MaxGroupGnostic, Step};
use compress_utils::{execute_compute_shader, time_it, wgpu_utils, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use log::info;
use pollster::block_on;
use std::cmp::{max, min};
use std::fs;
use std::sync::Arc;
use wgpu::naga::Expression::Math;
use wgpu::{Device, Queue};
use wgpu_types::BufferAddress;

#[async_trait]
impl Decompressor for BatchedGPUDecompressor {
    async fn decompress(
        &self,
        compressed_bytes_vec: &mut Vec<u8>,
    ) -> Result<Vec<f32>, DecompressionError> {
        let mut current_index = 0usize;
        let uncompressed_values;
        let mut total_millis = 0;
        time_it!(
            {
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < compressed_bytes_vec.len() {
                    let buffer_value_count = u8::from_be_bytes(
                        compressed_bytes_vec[current_index..current_index + size_of::<u8>()]
                            .try_into()
                            .unwrap(),
                    ) as usize
                        + 1;
                    current_index += size_of::<u8>();

                    let size_in_bytes = u32::from_be_bytes(
                        compressed_bytes_vec[current_index..current_index + size_of::<u32>()]
                            .try_into()
                            .unwrap(),
                    );
                    current_index += size_of::<u32>();
                    let byte_window_vec = compressed_bytes_vec
                        [current_index..current_index + (size_in_bytes as usize)]
                        .to_vec();
                    let mut byte_window = byte_window_vec.as_slice();
                    assert_eq!(
                        byte_window.len() % 4,
                        0,
                        "Total bytes need to be in batches of 4"
                    );
                    while let Some((first_four_bytes, rest)) = byte_window.split_at_checked(4) {
                        byte_window = rest;
                        //parse u32 from groups of 4 bytes
                        vec_window.push(*bytemuck::from_bytes::<u32>(first_four_bytes))
                    }
                    input_indexes.push(vec_window.len() as u32);
                    current_index += size_in_bytes as usize;
                    total_uncompressed_values += buffer_value_count
                }

                //Block is as many buffers fit into the gpu the distinction is made for compatibility reasons
                let block_values = self
                    .decompress_block(
                        vec_window.as_slice(),
                        input_indexes.as_slice(),
                        get_buffer_size(),
                    )
                    .await?;

                uncompressed_values = block_values[0..total_uncompressed_values]
                    .iter()
                    .copied()
                    .collect_vec();
            },
            total_millis,
            "decompression"
        );
        Ok(uncompressed_values)
    }
}

pub struct BatchedGPUDecompressor {
    context: Arc<Context>,
}
impl MaxGroupGnostic for BatchedGPUDecompressor {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len
    }
}

impl BatchedGPUDecompressor {
    pub(crate) async fn decompress_block(
        &self,
        compressed_value_slice: &[u32],
        input_indexes: &[u32],
        buffer_value_count: usize,
    ) -> Result<Vec<f32>, DecompressionError> {
        let loop_size = get_buffer_size().to_string();
        let shader_code = include_str!("shaders/decompress.wgsl").replace("@@size", &loop_size);

        //input_indexes shows how many buffers of count buffer_value_count, so we use workgroups equal to as many fit in the gpu
        let workgroup_count = min(self.get_max_number_of_groups(input_indexes.len()), 1);
        let mut result = Vec::new();
        info!("The wgpu workgroup size: {}", &workgroup_count);

        for iteration in 0..(input_indexes.len() / workgroup_count) {
            let iteration_compressed_values = compressed_value_slice
                [iteration * workgroup_count..(iteration + 1) * workgroup_count]
                .to_vec();
            let iteration_input_indexes = input_indexes
                [iteration * workgroup_count..(iteration + 1) * workgroup_count]
                .to_vec();

            info!(
                "The size in bytes of the compressed input vec: {}",
                iteration_compressed_values.len() * size_of::<u8>()
            );

            let out_buffer_size = buffer_value_count * size_of::<u32>();
            info!(
                "The uncompressed output values buffer size in bytes: {}",
                out_buffer_size
            );

            let input_storage_buffer = BufferWrapper::storage_with_content(
                self.device(),
                bytemuck::cast_slice(compressed_value_slice),
                WgpuGroupId::new(0, 1),
                Some("Storage input Buffer"),
            );
            let out_staging = BufferWrapper::stage_with_size(
                self.device(),
                out_buffer_size as BufferAddress,
                Some("Staging output Buffer"),
            );
            let out_storage_buffer = BufferWrapper::storage_with_size(
                self.device(),
                out_buffer_size as BufferAddress,
                WgpuGroupId::new(0, 0),
                Some("Storage output Buffer"),
            );

            let size_uniform = BufferWrapper::uniform_with_content(
                self.device(),
                bytemuck::bytes_of(&buffer_value_count),
                WgpuGroupId::new(0, 2),
                Some("Total input values"),
            );
            info!("Total output values: {}", buffer_value_count);
            let in_size = BufferWrapper::storage_with_content(
                self.device(),
                bytemuck::cast_slice(&iteration_input_indexes),
                WgpuGroupId::new(0, 3),
                Some("Total bytes input"),
            );
            info!("Total input values: {}", buffer_value_count);

            execute_compute_shader!(
                self.context(),
                &shader_code,
                vec![
                    &out_storage_buffer,
                    &input_storage_buffer,
                    &out_staging,
                    &size_uniform,
                    &in_size,
                ],
                workgroup_count
            );

            let output = wgpu_utils::get_s_output::<f32>(
                self.context(),
                out_storage_buffer.buffer(),
                (buffer_value_count * size_of::<f32>()) as BufferAddress,
                out_staging.buffer(),
            )
            .await?;
            result.extend(output);
        }
        info!("Output result size: {}", result.len());
        if trace_steps().contains(&Step::Decompress) {
            let trace_path = Step::Decompress.get_trace_file();
            let mut trace_output = String::new();

            result
                .iter()
                .for_each(|it| trace_output.push_str(it.to_string().as_str()));

            fs::write(&trace_path, trace_output)
                .map_err(|it| DecompressionError::FromBaseAnyhowError(anyhow::anyhow!(it)))?;
        }
        Ok(result)
    }

    pub fn new(context_builder: impl ToContextable) -> Self {
        Self {
            context: context_builder.context(),
        }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn device(&self) -> &Device {
        self.context.device()
    }
    pub fn queue(&self) -> &Queue {
        self.context.queue()
    }
}
pub trait ToContextable {
    fn context(self) -> Arc<Context>;
}

impl ToContextable for Arc<Context> {
    fn context(self) -> Arc<Context> {
        self
    }
}

impl ToContextable for Context {
    fn context(self) -> Arc<Context> {
        Arc::new(self)
    }
}

impl ToContextable for () {
    fn context(self) -> Arc<Context> {
        Arc::new(block_on(Context::initialize_default_adapter()).unwrap())
    }
}
