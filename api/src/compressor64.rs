use async_trait::async_trait;
use compress_utils::cpu_compress::{CompressionError, Compressor};

pub enum ChimpCompressor64 {
    // CPUCompressor(CPUCompressor)
    // GPUCompressor
}

#[async_trait]
impl<T> Compressor<f64> for ChimpCompressor64 {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError> {
        match self {
            _ => todo!("add in 64 bit support"),
        }
    }
}

// impl<T> From<T> for ChimpCompressor64<T>
// where
//     T: Compressor<f64>,
// {
//     fn from(value: T) -> Self {
//         Self { compressor: value }
//     }
// }
