use async_trait::async_trait;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};
use compress_utils::general_utils::DecompressResult;
use wgpu_compress_64_batched::decompressor::ChimpDecompressorBatched64;

pub enum ChimpDecompressor64 {
    CPUDeCompressor(ChimpDecompressorBatched64),
    GPUDeCompressor(ChimpDecompressorBatched64),
}

#[async_trait]
impl Decompressor<f64> for ChimpDecompressor64 {
    async fn decompress(
        &self,
        vec: &mut Vec<u8>,
    ) -> Result<DecompressResult<f64>, DecompressionError> {
        match self {
            ChimpDecompressor64::CPUDeCompressor(decompressor) => {
                decompressor.decompress(vec).await
            }
            ChimpDecompressor64::GPUDeCompressor(decompressor) => {
                decompressor.decompress(vec).await
            }
        }
    }
}
impl From<ChimpDecompressorBatched64> for ChimpDecompressor64 {
    fn from(value: ChimpDecompressorBatched64) -> Self {
        ChimpDecompressor64::GPUDeCompressor(value)
    }
}

// impl From<ChimpDecompressorBatched64> for ChimpDecompressor64 {
//     fn from(value: ChimpDecompressorBatched64) -> Self {
//         ChimpDecompressor64::CPUDeCompressor(value)
//     }
// }
