use async_trait::async_trait;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use wgpu_compress_32_batched::cpu::decompressor::BatchedDecompressorCpu;
use wgpu_compress_32_batched::decompressor::BatchedGPUDecompressor;
use wgpu_compress_64_batched::ChimpDecompressorBatched64;

pub enum ChimpDecompressor64 {
    CPUDeCompressor(ChimpDecompressorBatched64<BatchedDecompressorCpu>),
    GPUDeCompressor(ChimpDecompressorBatched64<BatchedGPUDecompressor>),
}

#[async_trait]
impl<T> Decompressor<f64> for ChimpDecompressor64 {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
        match self {
            ChimpDecompressor64::CPUDeCompressor(decompressor) => decompressor.decompress(vec),
            ChimpDecompressor64::GPUDeCompressor(decompressor) => decompressor.decompress(vec),
        }
    }
}
impl From<ChimpDecompressorBatched64<BatchedGPUDecompressor>> for ChimpDecompressor64 {
    fn from(value: ChimpDecompressorBatched64<BatchedGPUDecompressor>) -> Self {
        ChimpDecompressor64::GPUDeCompressor(value)
    }
}

impl From<ChimpDecompressorBatched64<BatchedDecompressorCpu>> for ChimpDecompressor64 {
    fn from(value: ChimpDecompressorBatched64<BatchedDecompressorCpu>) -> Self {
        ChimpDecompressor64::CPUDeCompressor(value)
    }
}
