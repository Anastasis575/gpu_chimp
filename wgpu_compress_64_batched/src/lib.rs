use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::cpu_compress::{
    CompressionError, Compressor, DecompressionError, Decompressor,
};
use itertools::Itertools;
use pollster::FutureExt;
use std::string::ToString;
use std::sync::Arc;
use wgpu_compress_32_batched::decompressor::BatchedGPUDecompressor;
use wgpu_compress_32_batched::{ChimpCompressorBatched, FinalizerEnum};

#[derive(Debug)]
struct ChimpCompressorBatched64<T>
where
    T: Compressor<f32>,
{
    compressor32bit: T,
}

impl ChimpCompressorBatched64<ChimpCompressorBatched> {
    pub fn new(debug: bool, context: Arc<Context>, finalizer: FinalizerEnum) -> Self {
        Self {
            compressor32bit: ChimpCompressorBatched::new(debug, context, finalizer),
        }
    }
}
impl Default for ChimpCompressorBatched64<ChimpCompressorBatched> {
    fn default() -> Self {
        Self {
            compressor32bit: ChimpCompressorBatched::default(),
        }
    }
}

// fn splitter(value: f64) -> [f32; 2] {
//     const LOW_MASK: u64 = 0xFFFF_FFFF;
//     const SHIFT: u32 = 32;
//
//     let bits = bytemuck::cast::<f64, u64>(value);
//     let high_bits = (bits >> SHIFT) as u32;
//     let low_bits = (bits & LOW_MASK) as u32;
//     [high_bits as f32, low_bits as f32]
// }
// fn merger(value: [f32; 2]) -> f64 {
//     let high = value[0] as u64;
//     let low = value[1] as u64;
//     let bits = (high << 32) | low;
//     bytemuck::cast::<u64, f64>(bits)
// }

fn splitter(value: f64) -> [f32; 2] {
    let bits = bytemuck::cast::<f64, u64>(value);
    [
        f32::from_bits((bits >> 32) as u32),
        f32::from_bits(bits as u32),
    ]
}
fn merger(value: [f32; 2]) -> f64 {
    let high = (value[0].to_bits() as u64) << 32;
    let low = value[1].to_bits() as u64;
    bytemuck::cast(high | low)
}

#[async_trait]
impl<T: Compressor<f32> + Send + Sync> Compressor<f64> for ChimpCompressorBatched64<T> {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError> {
        let mut split = vec.iter().map(|it| splitter(*it)).collect_vec();
        let mut final_values = split.iter_mut().map(|x| x[0]).collect_vec();
        let split_right_side = split.iter_mut().map(|x| x[1]).collect_vec();
        final_values.extend(split_right_side);
        let compressed = self.compressor32bit.compress(&mut final_values).await?;
        Ok(compressed)
    }
}

#[derive(Debug)]
struct ChimpDecompressorBatched64<T>
where
    T: Decompressor<f32>,
{
    decompressor32bits: T,
}

impl ChimpDecompressorBatched64<BatchedGPUDecompressor> {
    pub fn new(context: Arc<Context>) -> Self {
        Self {
            decompressor32bits: BatchedGPUDecompressor::new(context),
        }
    }
}
impl Default for ChimpDecompressorBatched64<BatchedGPUDecompressor> {
    fn default() -> Self {
        Self {
            decompressor32bits: BatchedGPUDecompressor::default(),
        }
    }
}

#[async_trait]
impl<T: Decompressor<f32> + Send + Sync> Decompressor<f64> for ChimpDecompressorBatched64<T> {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError> {
        let decompressed = self.decompressor32bits.decompress(vec).await?;
        assert_eq!(decompressed.len() % 2, 0);
        let mut merged = Vec::with_capacity(decompressed.len() / 2);
        let half = decompressed.len() / 2;
        (0..decompressed.len()).for_each(|i| {
            let values = [decompressed[i], decompressed[half + i]];
            merged.push(merger(values));
        });
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use crate::{merger, splitter};

    #[test]
    fn splitter_merger() {
        let original = 123.456789_f64;
        let split = splitter(original);
        let merged = merger(split);
        assert_eq!(original, merged);
    }

    #[test]
    fn splitter_merger2() {
        let original = 25.0003;
        let split = splitter(original);
        let merged = merger(split);
        assert_eq!(original, merged);
    }

    #[test]
    fn halfer() {
        let u64 = [1, 2, 3, 4, 5, 6];
        let xm = &u64[..u64.len() / 2];
        let mx = &u64[u64.len() / 2..];
        assert_eq!(xm.len(), mx.len());
    }
    #[test]
    fn eh() {
        let u64 = Vec::<i32>::with_capacity(5);
        assert_eq!(u64.len(), 5);
    }
}
