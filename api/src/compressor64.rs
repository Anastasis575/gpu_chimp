use async_trait::async_trait;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use compress_utils::general_utils::CompressResult;
use wgpu_compress_32_batched::ChimpCompressorBatched;
use wgpu_compress_64_batched::ChimpCompressorBatched64;

pub enum ChimpCompressor64 {
    GPUBatchCompressor64(ChimpCompressorBatched64),
}

#[async_trait]
impl Compressor<f64> for ChimpCompressor64 {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<CompressResult, CompressionError> {
        match self {
            ChimpCompressor64::GPUBatchCompressor64(compress) => compress.compress(vec).await,
        }
    }
}

impl From<ChimpCompressorBatched64> for ChimpCompressor64 {
    fn from(value: ChimpCompressorBatched64) -> Self {
        ChimpCompressor64::GPUBatchCompressor64(value)
    }
}
