use crate::calculate_indexes::{CalculateIndexes, GPUCalculateIndexes};
use crate::compute_s_shader::{ComputeS, ComputeSNImpl};
use crate::cpu;
use crate::final_compress::{FinalCompress, FinalCompressImpl};
use crate::finalize::{Finalize, Finalizer};
use crate::previous_indexes::{PreviousIndexes, PreviousIndexesNImpl};
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::{
    add_padding_to_fit_buffer_count, ChimpBufferInfo, CompressResult, Padding,
};
use compress_utils::types::{ChimpOutput, S};
use compress_utils::{time_it, wgpu_utils};
use itertools::Itertools;
use log::info;
use pollster::FutureExt;
use std::sync::Arc;

#[derive(Debug)]
pub struct ChimpNGPUBatched {
    context: Arc<Context>,
    n: usize,
}

impl ChimpNGPUBatched {
    pub(crate) fn previous_index_factory(&self) -> Box<dyn PreviousIndexes + Send + Sync> {
        Box::new(PreviousIndexesNImpl::new(self.context.clone(), self.n))
        // Box::new(cpu::previous_indexes::PreviousIndexesNCPUImpl {
        //     context: self.context.clone(),
        //     n: self.n,
        // })
    }
    pub(crate) fn compute_finalize_factory(&self) -> Box<dyn Finalize + Send + Sync> {
        Box::new(Finalizer::new(self.context.clone()))
    }

    pub(crate) fn calculate_indexes_factory(&self) -> Box<dyn CalculateIndexes + Send + Sync> {
        Box::new(GPUCalculateIndexes::new(self.context.clone()))
    }

    pub(crate) fn compute_final_compress_factory(&self) -> Box<dyn FinalCompress + Send + Sync> {
        // Box::new(FinalCompressImpl::new(self.context.clone(), self.n))
        Box::new(cpu::compress::CPUBatchedNCompressImpl {
            context: self.context.clone(),
            n: self.n,
        })
    }

    pub(crate) fn compute_s_factory(&self) -> Box<dyn ComputeS + Send + Sync> {
        Box::new(ComputeSNImpl::new(self.context.clone(), self.n))
        // Box::new(cpu::compute_s::CPUBatchedNComputeSImpl {
        //     context: self.context.clone(),
        //     n: self.n,
        // })
    }

    pub(crate) fn split_by_max_gpu_buffer_size(&self, vec: &mut Vec<f32>) -> Vec<Vec<f32>> {
        let max = self.context.get_max_storage_buffer_size();
        let mut split_by = max / size_of::<S>() - ChimpBufferInfo::get().buffer_size(); //The most costly buffer
        while ((split_by + 10) * size_of::<S>()) as u64
            >= self.context.get_max_storage_buffer_size() as u64
            || ((split_by + 10) * size_of::<ChimpOutput>()) as u64
                >= self.context.get_max_storage_buffer_size() as u64
        {
            split_by -= ChimpBufferInfo::get().buffer_size();
        }
        let closest = split_by - split_by % ChimpBufferInfo::get().buffer_size();

        let x = vec.chunks(closest).map(|it| it.to_vec()).collect_vec();
        x
    }

    pub fn new(context: Arc<Context>, n: usize) -> Self {
        Self { context, n }
    }
}

#[async_trait]
impl Compressor<f32> for ChimpNGPUBatched {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<CompressResult, CompressionError> {
        let compute_s_impl = self.compute_s_factory();
        let final_compress_impl = self.compute_final_compress_factory();
        let calculate_indexes_impl = self.calculate_indexes_factory();
        let finalize_impl = self.compute_finalize_factory();
        let previous_index_impl = self.previous_index_factory();

        let iterations = self.split_by_max_gpu_buffer_size(vec);
        let mut byte_stream = Vec::new();
        let mut metadata = 0usize;
        let mut buffers = wgpu_utils::RunBuffers::default();
        let mut skip_time = 0u128;
        for iteration_values in iterations {
            let mut padding = Padding(0);
            let buffer_size = ChimpBufferInfo::get().buffer_size();
            let mut values = iteration_values;
            values = add_padding_to_fit_buffer_count(values, buffer_size, &mut padding);
            let mut total_millis: u128 = 0;
            // let mut indexes;
            let output_vec;
            time_it!(
                {
                    previous_index_impl
                        .calculate_previous_indexes(&mut values, &mut buffers, &mut skip_time)
                        .await?;
                },
                total_millis,
                "calculation of the previous value to compare stage"
            );
            time_it!(
                {
                    compute_s_impl
                        .compute_s(&mut values, &mut buffers, &mut skip_time)
                        .await?;
                },
                total_millis,
                "s computation stage"
            );
            time_it!(
                {
                    final_compress_impl
                        .final_compress(&mut buffers, &mut skip_time)
                        .await?;
                },
                total_millis,
                "final output stage"
            );
            time_it!(
                {
                    calculate_indexes_impl
                        .calculate_indexes(
                            &mut buffers,
                            ChimpBufferInfo::get().buffer_size() as u32,
                            &mut skip_time,
                        )
                        .await?;
                },
                total_millis,
                "final output stage"
            );
            time_it!(
                {
                    output_vec = finalize_impl
                        .finalize(&mut buffers, padding.0, &mut skip_time)
                        .await?;
                },
                total_millis,
                "final Result collection"
            );
            byte_stream.extend(output_vec.compressed_value_ref());
            metadata += output_vec.metadata_size()
        }

        Ok(CompressResult(byte_stream, metadata, skip_time))
    }
}

impl Default for ChimpNGPUBatched {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            n: 128,
        }
    }
}
