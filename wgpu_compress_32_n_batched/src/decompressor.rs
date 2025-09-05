use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::{
    trace_steps, ChimpBufferInfo, DecompressResult, MaxGroupGnostic, Step,
};
use compress_utils::{
    execute_compute_shader, step, time_it, wgpu_utils, BufferWrapper, WgpuGroupId,
};
use log::info;
use pollster::FutureExt;
use std::cmp::{max, min};
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use wgpu::{Device, Queue};
use wgpu_types::BufferAddress;
#[async_trait]
impl Decompressor<f32> for BatchedGPUNDecompressor {
    async fn decompress(
        &self,
        compressed_bytes_vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f32>, DecompressionError> {
        let mut current_index = 0usize;
        let mut uncompressed_values = Vec::new();
        let mut total_millis = 0;
        let mut skip_time = 0;
        time_it!(
            {
                let compressed_bytes_vec: &[u32] = bytemuck::cast_slice(&compressed_bytes_vec);
                let mut vec_window = Vec::new();
                let mut total_uncompressed_values = 0;
                let mut input_indexes = Vec::new();
                while current_index < compressed_bytes_vec.len() {
                    while current_index < compressed_bytes_vec.len() {
                        let old_index = current_index;
                        let buffer_value_count = compressed_bytes_vec[current_index] + 1;
                        current_index += 1;

                        let size_in_bytes = compressed_bytes_vec[current_index];
                        let size = size_in_bytes as usize / size_of::<u32>();
                        current_index += 1;
                        if (vec_window.len() + size as usize) * size_of::<u32>() as usize
                            >= self.context.get_max_storage_buffer_size()
                        {
                            current_index = old_index;
                            break;
                        }

                        if (input_indexes.len() + 1) * ChimpBufferInfo::get().buffer_size() * 4
                            >= self.context.get_max_storage_buffer_size()
                        {
                            current_index = old_index;
                            break;
                        }

                        vec_window.extend(
                            compressed_bytes_vec[current_index..current_index + (size as usize)]
                                .to_vec(),
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
                            ChimpBufferInfo::get().buffer_size(),
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
            uncompressed_values
                .iter()
                .map(|it: &f32| it.to_string())
                .into_iter()
        });
        Ok(DecompressResult(uncompressed_values, skip_time))
    }
}

pub struct BatchedGPUNDecompressor {
    context: Arc<Context>,
    n: usize,
}
impl MaxGroupGnostic for BatchedGPUNDecompressor {
    fn get_max_number_of_groups(&self, _content_len: usize) -> usize {
        self.context().get_max_workgroup_size()
    }
}
impl Default for BatchedGPUNDecompressor {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            n: 128,
        }
    }
}
impl BatchedGPUNDecompressor {
    pub(crate) async fn decompress_block(
        &self,
        compressed_value_slice: &[u32],
        input_indexes: &[u32],
        buffer_value_count: usize,
        skip_time: &mut u128,
    ) -> Result<Vec<f32>, DecompressionError> {
        //how many buffers fit into the GPU
        let workgroup_count = self.get_max_number_of_groups(input_indexes.len());

        //how many iterations I need to fully decompress all the buffers
        let iterator_count = ((input_indexes.len() - 1) / workgroup_count) + 1;

        let instant = Instant::now();

        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(&compressed_value_slice),
            WgpuGroupId::new(0, 1),
            Some("Storage input Buffer"),
        );

        let in_size = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(&input_indexes),
            WgpuGroupId::new(0, 3),
            Some("Total bytes input"),
        );

        let out_buffer_size =
            (input_indexes.len() - 1) * ChimpBufferInfo::get().buffer_size() * size_of::<f32>();

        let out_staging = BufferWrapper::stage_with_size(
            self.device(),
            out_buffer_size as BufferAddress,
            Some("Staging output Buffer"),
        );
        let last_lead_storage = BufferWrapper::storage_with_size(
            self.device(),
            out_buffer_size as BufferAddress,
            WgpuGroupId::new(0, 5),
            Some("Last Lead storage Buffer"),
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
            // let out_offset = input_indexes[iteration * workgroup_count];
            // let next_out = input_indexes[next - 1];
            // let iteration_compressed_values_len = next_out - out_offset;

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
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                iteration * self.context.get_max_workgroup_size()
            );
            let n = format!("let n={}u;", self.n);
            let log2n = format!("let log2n={}u;", self.n.ilog2());
            let shader_code = include_str!("shaders/decompress.wgsl")
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@n", &n)
                .replace("//@log2n", &log2n)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &shader_code,
                vec![
                    &out_storage_buffer,
                    &last_lead_storage,
                    &input_storage_buffer,
                    &out_staging,
                    &size_uniform,
                    &in_size,
                    &input_size_uniform
                ],
                iteration_input_indexes - 1,
                Some("decompress pass")
            );
        }
        let instant = Instant::now();
        let result = wgpu_utils::get_from_gpu::<f32>(
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

    pub fn new(context_builder: impl Into<Arc<Context>>, n: usize) -> Self {
        Self {
            context: context_builder.into(),
            n,
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
