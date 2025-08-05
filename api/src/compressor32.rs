use async_trait::async_trait;
use compress_utils::cpu_compress::{CompressionError, Compressor};
use wgpu_compress_32_batched::ChimpCompressorBatched;

pub enum ChimpCompressor32 {
    // CPUCompressor32(CPUCompressor),
    // GPUCompressor32(ChimpCompressor),
    GPUBatchCompressor32(ChimpCompressorBatched),
}

#[async_trait]
impl Compressor<f32> for ChimpCompressor32 {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>, CompressionError> {
        match self {
            // ChimpCompressor32::CPUCompressor32(compressor) => compressor(vec).await,
            // ChimpCompressor32::GPUCompressor32(compressor) => compressor(vec).await,
            ChimpCompressor32::GPUBatchCompressor32(compressor) => compressor.compress(vec).await,
        }
    }
}
// impl From<CPUCompressor> for ChimpCompressor32 {
//     fn from(value: CPUCompressor) -> Self {
//         ChimpCompressor32::CPUCompressor32(value)
//     }
// }
//
// impl From<ChimpCompressor> for ChimpCompressor32 {
//     fn from(value: ChimpCompressor) -> Self {
//         ChimpCompressor32::GPUCompressor32(value)
//     }
// }
impl From<ChimpCompressorBatched> for ChimpCompressor32 {
    fn from(value: ChimpCompressorBatched) -> Self {
        ChimpCompressor32::GPUBatchCompressor32(value)
    }
}
