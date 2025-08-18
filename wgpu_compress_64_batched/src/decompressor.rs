use crate::cpu::decompressor::CPUDecompressorBatched64;
use crate::ChimpCompressorBatched64;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::DeviceEnum::GPU;
use compress_utils::general_utils::{
    trace_steps, ChimpBufferInfo, DeviceEnum, MaxGroupGnostic, Step,
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
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
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
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
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
    ) -> Result<Vec<f64>, DecompressionError> {
        let mut current_index = 0usize;
        let mut uncompressed_values = Vec::new();
        let mut total_millis = 0;
        time_it!(
            {
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < compressed_bytes_vec.len() {
                    while current_index < compressed_bytes_vec.len() {
                        let old_index = current_index;
                        let buffer_value_count = u32::from_le_bytes(
                            compressed_bytes_vec[current_index..current_index + size_of::<u32>()]
                                .try_into()
                                .unwrap(),
                        ) as usize
                            + 1;
                        current_index += size_of::<u32>();
                        let size_in_bytes = u32::from_le_bytes(
                            compressed_bytes_vec[current_index..current_index + size_of::<u32>()]
                                .try_into()
                                .unwrap(),
                        );
                        if vec_window.len() + size_in_bytes as usize
                            >= ChimpCompressorBatched64::MAX_BUFFER_SIZE_BYTES / size_of::<u64>()
                        {
                            current_index = old_index;
                            break;
                        }
                        current_index += size_of::<u32>();

                        let byte_window_vec = compressed_bytes_vec
                            [current_index..current_index + (size_in_bytes as usize)]
                            .to_vec();
                        let mut byte_window = byte_window_vec.as_slice();
                        assert_eq!(
                            byte_window.len() % 8,
                            0,
                            "Total bytes need to be in batches of 8"
                        );
                        while let Some((first_four_bytes, rest)) = byte_window.split_at_checked(8) {
                            byte_window = rest;
                            //parse u32 from groups of 8 bytes
                            let value_u64 =
                                u64::from_le_bytes(first_four_bytes.try_into().unwrap());
                            vec_window.push(value_u64);
                        }
                        input_indexes.push(vec_window.len() as u32);
                        current_index += size_in_bytes as usize;
                        total_uncompressed_values += buffer_value_count
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
                        )
                        .await?;

                    uncompressed_values.extend(block_values[0..total_uncompressed_values].iter());
                }
            },
            total_millis,
            "decompression"
        );
        Ok(uncompressed_values)
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
    ) -> Result<Vec<f64>, DecompressionError> {
        let util_64 = include_str!("shaders/64_utils.wgsl");

        let shader_code = include_str!("shaders/decompress.wgsl")
            .replace("//#include(64_utils)", util_64)
            .to_string();

        //how many buffers fit into the GPU
        let workgroup_count = self.get_max_number_of_groups(input_indexes.len());

        //how many iterations I need to fully decompress all the buffers
        let iterator_count = max((input_indexes.len() - 1) / workgroup_count, 1);

        //input_indexes shows how many buffers of count buffer_value_count, so we use workgroups equal to as many fit in the gpu
        let mut result = Vec::<f64>::new();
        info!("The wgpu workgroup size: {}", &workgroup_count);

        for iteration in 0..iterator_count {
            //split all the buffers to the chunks each iteration will use
            let is_last_iteration = iteration == iterator_count - 1;
            let iteration_input_indexes = if is_last_iteration {
                input_indexes[iteration * workgroup_count..].to_vec()
            } else {
                input_indexes[iteration * workgroup_count..(iteration + 1) * workgroup_count]
                    .to_vec()
            };
            let first_index = iteration_input_indexes[0] as usize;
            let iteration_compressed_values = if is_last_iteration {
                compressed_value_slice[first_index..].to_vec()
            } else {
                compressed_value_slice
                    [first_index..(iteration_input_indexes.last().unwrap().to_owned() as usize)]
                    .to_vec()
            };

            info!(
                "The size in bytes of the compressed input vec: {}",
                iteration_compressed_values.len() * size_of::<u8>()
            );

            let out_buffer_size = (iteration_input_indexes.len() - 1)
                * ChimpBufferInfo::get().buffer_size()
                * size_of::<f64>();
            info!(
                "The uncompressed output values buffer size in bytes: {}",
                out_buffer_size
            );

            let input_storage_buffer = BufferWrapper::storage_with_content(
                self.device(),
                bytemuck::cast_slice(iteration_compressed_values.as_slice()),
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
            let input_size_uniform = BufferWrapper::uniform_with_content(
                self.device(),
                bytemuck::bytes_of(&iteration_compressed_values.len()),
                WgpuGroupId::new(0, 4),
                Some("Total input buffer length"),
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
                    &input_size_uniform
                ],
                iteration_input_indexes.len() - 1,
                Some("decompress pass")
            );

            let output = wgpu_utils::get_s_output::<f64>(
                self.context(),
                out_storage_buffer.buffer(),
                ((iteration_input_indexes.len() - 1) * buffer_value_count * size_of::<f64>())
                    as BufferAddress,
                out_staging.buffer(),
            )
            .await?;
            result.extend(output);
        }
        info!("Output result size: {}", result.len());
        step!(Step::Decompress, {
            result.iter().map(|it| it.to_string()).into_iter()
        });
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
