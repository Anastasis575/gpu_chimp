use crate::compute_s_shader::{ComputeS, ComputeSImpl};
use crate::cpu;
use crate::final_compress::{FinalCompress, FinalCompressImpl64};
use crate::finalize::{Finalize, Finalizer64};
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::{ChimpBufferInfo, DeviceEnum, MaxGroupGnostic, Padding};
use compress_utils::time_it;
use compress_utils::types::{ChimpOutput64, S};
use log::info;
use pollster::FutureExt;
use std::ops::Div;
use std::sync::Arc;

#[derive(Debug)]
pub struct ChimpCompressorBatched64 {
    // debug: bool,
    context: Arc<Context>,
    device_type: DeviceEnum,
}
impl Default for ChimpCompressorBatched64 {
    fn default() -> Self {
        Self {
            context: Arc::new(Context::initialize_default_adapter().block_on().unwrap()),
            device_type: DeviceEnum::GPU,
        }
    }
}

pub fn add_padding_to_fit_buffer_count_64(
    mut values: Vec<f64>,
    buffer_size: usize,
    padding: &mut Padding,
) -> Vec<f64> {
    if values.len() % buffer_size != 0 {
        let count = (values.len().div(buffer_size) + 1) * buffer_size - values.len();
        padding.0 = count;
        for _i in 0..count {
            values.push(0f64);
        }
    }
    values
}

#[async_trait]
impl Compressor<f64> for ChimpCompressorBatched64 {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError> {
        let mut padding = Padding(0);
        let buffer_size = ChimpBufferInfo::get().buffer_size();
        let mut values = vec.to_owned();
        values = add_padding_to_fit_buffer_count_64(values, buffer_size, &mut padding);
        let mut total_millis = 0;
        let mut s_values: Vec<S>;
        let mut chimp_vec: Vec<ChimpOutput64>;

        let compute_s_impl = self.compute_s_factory();
        let final_compress_impl = self.compute_final_compress_factory();
        let finalize_impl = self.compute_finalize_factory();

        let output_vec: BitVec;
        time_it!(
            {
                s_values = compute_s_impl.compute_s(&mut values).await?;
            },
            total_millis,
            "s computation stage"
        );
        time_it!(
            {
                chimp_vec = final_compress_impl
                    .final_compress(&mut values, &mut s_values, 0)
                    .await?;
            },
            total_millis,
            "final output stage"
        );
        time_it!(
            {
                output_vec = BitVec::from_bytes(
                    finalize_impl
                        .finalize(&mut chimp_vec, padding.0)
                        .await?
                        .as_slice(),
                );
            },
            total_millis,
            "final Result collection"
        );
        Ok(output_vec.to_bytes())
    }
}
enum ComputeS64Impls {
    GPU(ComputeSImpl),
    CPU(cpu::compute_s::CpuComputeSImpl),
}

impl MaxGroupGnostic for ComputeS64Impls {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        match self {
            ComputeS64Impls::GPU(c) => c.get_max_number_of_groups(content_len),
            ComputeS64Impls::CPU(c) => c.get_max_number_of_groups(content_len),
        }
    }
}

#[async_trait]
impl ComputeS for ComputeS64Impls {
    async fn compute_s(&self, values: &mut [f64]) -> anyhow::Result<Vec<S>> {
        match self {
            ComputeS64Impls::GPU(c) => c.compute_s(values).await,
            ComputeS64Impls::CPU(c) => c.compute_s(values).await,
        }
    }
}
enum Compress64Impls {
    GPU(FinalCompressImpl64),
    CPU(cpu::chimp_compress::CPUFinalCompressImpl64),
}

impl MaxGroupGnostic for Compress64Impls {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        match self {
            Compress64Impls::GPU(c) => c.get_max_number_of_groups(content_len),
            Compress64Impls::CPU(c) => c.get_max_number_of_groups(content_len),
        }
    }
}

#[async_trait]
impl FinalCompress for Compress64Impls {
    async fn final_compress(
        &self,
        input: &mut Vec<f64>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> anyhow::Result<Vec<ChimpOutput64>> {
        match self {
            Compress64Impls::GPU(c) => c.final_compress(input, s_values, padding).await,
            Compress64Impls::CPU(c) => c.final_compress(input, s_values, padding).await,
        }
    }
}

enum Finalizer64impls {
    GPU(Finalizer64),
    CPU(cpu::finalize::CPUFinalizer64),
}
#[async_trait]
impl Finalize for Finalizer64impls {
    async fn finalize(
        &self,
        chimp_output: &mut Vec<ChimpOutput64>,
        padding: usize,
    ) -> anyhow::Result<Vec<u8>> {
        match self {
            Finalizer64impls::GPU(f) => f.finalize(chimp_output, padding).await,
            Finalizer64impls::CPU(f) => f.finalize(chimp_output, padding).await,
        }
    }
}
#[allow(unused)]
impl ChimpCompressorBatched64 {
    fn compute_s_factory(&self) -> ComputeS64Impls {
        match self.device_type() {
            DeviceEnum::GPU => ComputeS64Impls::GPU(ComputeSImpl::new(self.context.clone())),
            DeviceEnum::CPU => {
                ComputeS64Impls::CPU(cpu::compute_s::CpuComputeSImpl::new(self.context.clone()))
            }
        }
    }
    fn compute_final_compress_factory(&self) -> Compress64Impls {
        match self.device_type() {
            &DeviceEnum::GPU => {
                Compress64Impls::GPU(FinalCompressImpl64::new(self.context.clone(), false))
            }
            &DeviceEnum::CPU => Compress64Impls::CPU(
                cpu::chimp_compress::CPUFinalCompressImpl64::new(self.context.clone(), false),
            ),
        }
    }
    fn compute_finalize_factory(&self) -> Finalizer64impls {
        match self.device_type() {
            &DeviceEnum::GPU => Finalizer64impls::GPU(Finalizer64::new(self.context.clone())),
            &DeviceEnum::CPU => {
                Finalizer64impls::CPU(cpu::finalize::CPUFinalizer64::new(self.context.clone()))
            }
        }
    }
    pub(crate) fn new(context: impl Into<Arc<Context>>) -> Self {
        Self {
            context: context.into(),
            device_type: DeviceEnum::GPU,
        }
    }

    pub(crate) fn with_device(self, device: impl Into<DeviceEnum>) -> Self {
        Self {
            device_type: device.into(),
            ..self
        }
    }
    pub(crate) fn device_type(&self) -> &DeviceEnum {
        &self.device_type
    }
}
