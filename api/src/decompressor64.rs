use async_trait::async_trait;
use compress_utils::cpu_compress::{DecompressionError, Decompressor};

pub enum ChimpDecompressor64 {
    // CPUCompressor(CPUCompressor),
    // GPUCOmpressor(GPUCompressor
}

#[async_trait]
impl<T> Decompressor<f64> for ChimpDecompressor64 {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
        match self {
            _ => todo!("add in 64 bit support branch"),
        }
    }
}
