use async_trait::async_trait;
use compress_utils::cpu_compress::{CPUCompressor, DecompressionError, Decompressor};
use compress_utils::general_utils::DecompressResult;
use wgpu_compress_32_batched::decompressor::BatchedGPUDecompressor;

pub enum ChimpDecompressor32 {
    ChimpDecompressor32(CPUCompressor),
    //GPUDecompressor32,
    GPUBatchedDecompressor32(BatchedGPUDecompressor),
}

#[async_trait]
impl Decompressor<f32> for ChimpDecompressor32 {
    async fn decompress(
        &self,
        vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f32>, DecompressionError> {
        match self {
            ChimpDecompressor32::ChimpDecompressor32(decompressor) => {
                decompressor.decompress(vec).await
            }
            ChimpDecompressor32::GPUBatchedDecompressor32(decompressor) => {
                decompressor.decompress(vec).await
            }
        }
    }
}

impl From<CPUCompressor> for ChimpDecompressor32 {
    fn from(value: CPUCompressor) -> Self {
        ChimpDecompressor32::ChimpDecompressor32(value)
    }
}

impl From<BatchedGPUDecompressor> for ChimpDecompressor32 {
    fn from(value: BatchedGPUDecompressor) -> Self {
        ChimpDecompressor32::GPUBatchedDecompressor32(value)
    }
}
