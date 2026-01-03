use crate::cpu::decompressor::CPUDecompressorBatched64;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::DeviceEnum::GPU;
use compress_utils::general_utils::{
    trace_steps, ChimpBufferInfo, DecompressResult, DeviceEnum, MaxGroupGnostic, Step,
};
use compress_utils::{
    execute_compute_shader, step, time_it, wgpu_utils, BufferWrapper, WgpuGroupId,
};
use itertools::Itertools;
use log::info;
use pollster::FutureExt;
use std::cmp::{max, min};
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use wgpu::{Device, Queue};
use wgpu_types::BufferAddress;
pub enum DecompressorImpl {
    GPU(GPUDecompressorBatched64),
    CPU(CPUDecompressorBatched64),
}

impl MaxGroupGnostic for DecompressorImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        match self {
            DecompressorImpl::GPU(d) => d.get_max_number_of_groups(content_len),
            DecompressorImpl::CPU(d) => d.get_max_number_of_groups(content_len),
        }
    }
}

#[async_trait]
impl Decompressor<f64> for DecompressorImpl {
    async fn decompress(
        &self,
        vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f64>, DecompressionError> {
        match self {
            DecompressorImpl::GPU(d) => d.decompress(vec).await,
            DecompressorImpl::CPU(d) => d.decompress(vec).await,
        }
    }
}

pub struct ChimpDecompressorBatched64 {
    context: Arc<Context>,
    device_type: DeviceEnum,
}
impl Default for ChimpDecompressorBatched64 {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            device_type: GPU,
        }
    }
}
impl MaxGroupGnostic for ChimpDecompressorBatched64 {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        self.decompressor_factory()
            .get_max_number_of_groups(content_len)
    }
}
#[async_trait]
impl Decompressor<f64> for ChimpDecompressorBatched64 {
    async fn decompress(
        &self,
        vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f64>, DecompressionError> {
        self.decompressor_factory().decompress(vec).await
    }
}
impl ChimpDecompressorBatched64 {
    pub fn new(context: Arc<Context>) -> Self {
        Self {
            context,
            device_type: GPU,
        }
    }
    pub fn with_device(self, device: impl Into<DeviceEnum>) -> Self {
        Self {
            device_type: device.into(),
            ..self
        }
    }

    pub fn decompressor_factory(&self) -> DecompressorImpl {
        match self.device_type() {
            GPU => DecompressorImpl::GPU(GPUDecompressorBatched64::new(self.context.clone())),
            DeviceEnum::CPU => {
                DecompressorImpl::CPU(CPUDecompressorBatched64::new(self.context.clone()))
            }
        }
    }

    pub fn device_type(&self) -> &DeviceEnum {
        &self.device_type
    }
}

#[async_trait]
impl Decompressor<f64> for GPUDecompressorBatched64 {
    async fn decompress(
        &self,
        compressed_bytes_vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f64>, DecompressionError> {
        let mut current_index = 0usize;
        let mut uncompressed_values = Vec::new();
        let mut total_millis = 0;
        let mut skip_time = 0;
        time_it!(
            {
                let mut compressed_bytes_vec: &[u64] = bytemuck::cast_slice(compressed_bytes_vec);
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < compressed_bytes_vec.len() {
                    while current_index < compressed_bytes_vec.len() {
                        let old_index = current_index;
                        let size_in_bytes =
                            (compressed_bytes_vec[current_index] & 0xFFFFFFFF) as usize;
                        let size = size_in_bytes / size_of::<u64>();

                        if (vec_window.len() + size_in_bytes as usize)
                            >= self.context.get_max_storage_buffer_size() / size_of::<u64>()
                        {
                            current_index = old_index;
                            break;
                        }
                        let buffer_value_count = (compressed_bytes_vec[current_index] >> 32) + 1;
                        current_index += 1;
                        if (input_indexes.len() + 1) * ChimpBufferInfo::get().buffer_size() * 4
                            >= self.context.get_max_storage_buffer_size()
                        {
                            current_index = old_index;
                            break;
                        }
                        vec_window.extend(
                            compressed_bytes_vec[current_index..current_index + size].to_vec(),
                        );

                        input_indexes.push(vec_window.len() as u32);
                        current_index += size as usize;
                        total_uncompressed_values += buffer_value_count as usize
                    }
                    input_indexes.insert(0, 0);
                    //Block is as many buffers fit into the gpu the distinction is made for compatibility reasons
                    let block_values = self
                        .decompress_block(
                            vec_window.as_slice(),
                            input_indexes.as_slice(),
                            min(
                                total_uncompressed_values,
                                ChimpBufferInfo::get().buffer_size(),
                            ),
                            &mut skip_time,
                        )
                        .await?;

                    uncompressed_values.extend(block_values[0..total_uncompressed_values].iter());
                    vec_window.clear();
                    total_uncompressed_values = 0;
                    input_indexes.clear();
                }
            },
            total_millis,
            "decompression"
        );
        step!(Step::Decompress, {
            uncompressed_values.iter().map(|it: &f64| it.to_string())
        });
        Ok(DecompressResult(uncompressed_values, skip_time))
    }
}

pub struct GPUDecompressorBatched64 {
    context: Arc<Context>,
}
impl MaxGroupGnostic for GPUDecompressorBatched64 {
    fn get_max_number_of_groups(&self, _content_len: usize) -> usize {
        self.context().get_max_workgroup_size()
    }
}
impl Default for GPUDecompressorBatched64 {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
        }
    }
}
impl GPUDecompressorBatched64 {
    pub(crate) async fn decompress_block(
        &self,
        compressed_value_slice: &[u64],
        input_indexes: &[u32],
        buffer_value_count: usize,
        skip_time: &mut u128,
    ) -> Result<Vec<f64>, DecompressionError> {
        //how many buffers fit into the GPU
        let workgroup_count = self.get_max_number_of_groups(input_indexes.len()) * 256;

        //how many iterations I need to fully decompress all the buffers
        let iterator_count = ((input_indexes.len() - 1) / workgroup_count) + 1;

        //input_indexes shows how many buffers of count buffer_value_count, so we use workgroups equal to as many fit in the gpu
        let mut result = Vec::<f64>::new();
        //info!("The wgpu workgroup size: {}", &workgroup_count);
        let instant = Instant::now();
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(&compressed_value_slice),
            WgpuGroupId::new(0, 1),
            Some("Storage input Buffer"),
        );
        //info!("Total output values: {}", buffer_value_count);
        let in_size = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(&input_indexes),
            WgpuGroupId::new(0, 3),
            Some("Total bytes input"),
        );
        let out_buffer_size =
            (input_indexes.len() - 1) * ChimpBufferInfo::get().buffer_size() * size_of::<f64>();
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
        *skip_time += instant.elapsed().as_millis();
        let workgroup_count = min(workgroup_count, self.context.get_max_workgroup_size());
        for iteration in 0..iterator_count {
            let util_64 = include_str!("shaders/64_utils.wgsl");
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                iteration * self.context.get_max_workgroup_size()
            );

            //split all the buffers to the chunks each iteration will use
            let is_last_iteration = iteration == iterator_count - 1;
            let offset = iteration * workgroup_count;
            let next = if is_last_iteration {
                input_indexes.len()
            } else {
                ((iteration + 1) * workgroup_count) + 1
            };
            let iteration_input_indexes = next - offset;
            // let iteration_input_indexes = if is_last_iteration {
            //     input_indexes[iteration * workgroup_count..].to_vec()
            // } else {
            //     input_indexes[iteration * workgroup_count..(iteration + 1) * workgroup_count]
            //         .to_vec()
            // };
            let out_offset = input_indexes[iteration * workgroup_count];
            let next_out = input_indexes[next - 1];
            let iteration_compressed_values_len = next_out - out_offset;

            let size_uniform = BufferWrapper::uniform_with_content(
                self.device(),
                bytemuck::bytes_of(&buffer_value_count),
                WgpuGroupId::new(0, 2),
                Some("Total input values"),
            );

            let input_size_uniform = BufferWrapper::uniform_with_content(
                self.device(),
                bytemuck::bytes_of(&compressed_value_slice.len()),
                WgpuGroupId::new(0, 4),
                Some("Total input buffer length"),
            );

            let s_offset = format!("let in_offset={offset}u;");
            let total_threads = format!("let total_threads={}u;", input_indexes.len());

            let shader_code = include_str!("shaders/decompress.wgsl")
                .replace("//#include(64_utils)", util_64)
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@total_threads", &total_threads)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &shader_code,
                vec![
                    &out_storage_buffer,
                    &input_storage_buffer,
                    &out_staging,
                    &size_uniform,
                    &in_size,
                    &input_size_uniform
                ],
                iteration_input_indexes.div_ceil(256),
                Some("decompress pass")
            );
        }

        let instant = Instant::now();
        let result = wgpu_utils::get_from_gpu::<f64>(
            self.context(),
            out_storage_buffer.buffer(),
            out_storage_buffer.size() as BufferAddress,
            out_staging.buffer(),
        )
        .await?;
        *skip_time += instant.elapsed().as_millis();
        //info!("Output result size: {}", result.len());
        Ok(result)
    }

    pub fn new(context_builder: impl Into<Arc<Context>>) -> Self {
        Self {
            context: context_builder.into(),
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
