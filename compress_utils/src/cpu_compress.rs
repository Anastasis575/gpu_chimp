use crate::bit_utils::{to_bit_vec, BitReadable, BitWritable};
use async_trait::async_trait;
use bit_vec::BitVec;
use thiserror::Error;

#[derive(Debug, Default, Clone)]
pub struct CPUCompressor {
    debug: bool,
}
#[derive(Error, Debug)]
pub enum CPUCompressError {
    #[error("Wrong format on decompression input at bit {index} in byte {byte_index}",byte_index=(.index)/8)]
    WrongFormat { index: usize },
}

#[derive(Error, Debug)]
pub enum CompressionError {
    #[error(transparent)]
    CpuCompressError(#[from] CPUCompressError),
    #[error(transparent)]
    FromBaseError(#[from] Box<dyn std::error::Error + Send>),
    #[error(transparent)]
    FromBaseAnyhowError(#[from] anyhow::Error),
}
#[derive(Error, Debug)]
pub enum DecompressionError {
    #[error(transparent)]
    CpuCompressError(#[from] CPUCompressError),
    #[error(transparent)]
    FromBaseAnyhowError(#[from] anyhow::Error),
}

impl CPUCompressor {
    pub fn new(debug: bool) -> Self {
        Self { debug }
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

#[async_trait]
pub trait Compressor {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>, CompressionError>;
}

#[async_trait]
pub trait Decompressor {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>, DecompressionError>;
}
#[async_trait]
pub trait Compressor64 {
    async fn compress(&self, vec: &mut Vec<f64>) -> Result<Vec<u8>, CompressionError>;
}

#[async_trait]
pub trait Decompressor64 {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f64>, DecompressionError>;
}
#[async_trait]
impl Compressor for CPUCompressor {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>, CompressionError> {
        let mut bit_vec = to_bit_vec(vec[0].to_bits());
        let mut last_lead = 0;
        for i in 1..vec.len() {
            let xorred = (vec[i].to_bits()) ^ (vec[i - 1].to_bits());
            let lead = xorred.leading_zeros() as usize;
            let trail = xorred.trailing_zeros() as usize;
            if trail > 6 {
                bit_vec.push(false);
                if xorred == 0 {
                    bit_vec.push(false);
                } else {
                    bit_vec.push(true);
                    bit_vec.write_bits(lead as u32, 5);
                    if lead + trail > 32 {
                        return Err(CompressionError::from(CPUCompressError::WrongFormat {
                            index: i,
                        }));
                    }
                    let center_bits = 32 - lead - trail;
                    bit_vec.write_bits(center_bits as u32, 5);
                    bit_vec.write_bits(xorred >> trail, center_bits as u32);
                }
            } else {
                bit_vec.push(true);
                if lead == last_lead {
                    bit_vec.push(false);
                    bit_vec.write_bits(xorred, 32 - lead as u32);
                } else {
                    bit_vec.push(true);
                    bit_vec.write_bits(lead as u32, 5);
                    bit_vec.write_bits(xorred, 32 - lead as u32);
                }
            }
            last_lead = lead;
        }
        Ok(bit_vec.to_bytes())
    }
}

#[async_trait]
impl Decompressor for CPUCompressor {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>, DecompressionError> {
        let input_vector = BitVec::from_bytes(vec.as_slice());
        let mut input_index: usize;
        let first_num_u32: u32 = input_vector.reinterpret_u32(0, 32);
        let first_num = f32::from_bits(first_num_u32);
        if self.debug {
            log::info!("0:{}", first_num);
        }

        let mut output = vec![first_num];

        let mut last_num = first_num.to_bits();
        let mut last_lead = 0;
        input_index = 32;
        while input_index < input_vector.len() {
            if input_index + 1 >= input_vector.len() {
                break;
            }
            if input_vector[input_index] {
                input_index += 1;
                let mut lead = last_lead;
                if input_vector[input_index] {
                    input_index += 1;
                    lead = input_vector.reinterpret_u32(input_index, 5);
                    input_index += 5;
                } else {
                    input_index += 1;
                }
                if lead > 32 {
                    return Err(DecompressionError::from(CPUCompressError::WrongFormat {
                        index: input_index,
                    }));
                }
                let mut significant_bits = 32 - lead;
                if significant_bits == 0 {
                    significant_bits = 32;
                }
                let value = input_vector.reinterpret_u32(input_index, significant_bits as usize);
                input_index += (32 - lead) as usize;
                let value = value ^ last_num;
                last_num = value;
                last_lead = lead;

                if value == u32::MAX {
                    break;
                } else {
                    let value_f32 = f32::from_bits(value);
                    if self.debug {
                        log::info!("{}:{}", output.len(), value_f32);
                    }
                    output.push(value_f32);
                }
            } else if input_vector[input_index + 1] {
                input_index += 2;
                let lead = input_vector.reinterpret_u32(input_index, 5);
                input_index += 5;
                let mut significant_bits = input_vector.reinterpret_u32(input_index, 5);
                input_index += 5;
                if significant_bits == 0 {
                    significant_bits = 32;
                }
                if lead + significant_bits > 32 {
                    return Err(DecompressionError::from(CPUCompressError::WrongFormat {
                        index: input_index,
                    }));
                }
                let trail = 32 - lead - significant_bits;
                let mut value =
                    input_vector.reinterpret_u32(input_index, (32 - lead - trail) as usize);
                input_index += (32 - lead - trail) as usize;
                value <<= trail;
                value ^= last_num;
                last_lead = lead;
                last_num = value;
                if value == u32::MAX {
                    break;
                } else {
                    let value_f32 = f32::from_bits(value);
                    if self.debug {
                        log::info!("{}:{}", output.len(), value_f32);
                    }
                    output.push(value_f32);
                }
            } else {
                let value_f32 = f32::from_bits(last_num);
                last_lead = 32;
                if self.debug {
                    log::info!("{}:{}", output.len(), value_f32);
                }
                output.push(value_f32);
                input_index += 2;
            }
        }

        Ok(output)
    }
}

pub struct TimedDecompressor<T>
where
    T: Decompressor + Send + Sync,
{
    decompressor: T,
}

impl From<CPUCompressor> for TimedDecompressor<CPUCompressor> {
    fn from(value: CPUCompressor) -> Self {
        Self {
            decompressor: value,
        }
    }
}

#[async_trait]
impl<T> Decompressor for TimedDecompressor<T>
where
    T: Decompressor + Send + Sync,
{
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>, DecompressionError> {
        let mut total_millis: u128 = 0;
        let times = std::time::Instant::now();
        log::info!("Started cpu decompression stage");
        log::info!("============================");
        let output = self.decompressor.decompress(vec).await?;
        log::info!("============================");
        log::info!("Finished cpu decompresssion stage");
        total_millis += times.elapsed().as_millis();
        log::info!("Stage execution time: {}ms", times.elapsed().as_millis());
        log::info!("Total time elapsed: {}ms", total_millis);
        log::info!("============================");
        Ok(output)
    }
}

pub struct TimedCompressor<T>
where
    T: Compressor + Send + Sync,
{
    compressor: T,
}
impl From<CPUCompressor> for TimedCompressor<CPUCompressor> {
    fn from(value: CPUCompressor) -> Self {
        Self { compressor: value }
    }
}
#[async_trait]
impl<T> Compressor for TimedCompressor<T>
where
    T: Compressor + Send + Sync,
{
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>, CompressionError> {
        let mut total_millis: u128 = 0;
        let times = std::time::Instant::now();
        log::info!("Started cpu compression stage");
        log::info!("============================");
        let output = self.compressor.compress(vec).await?;
        log::info!("============================");
        log::info!("Finished cpu compresion stage");
        total_millis += times.elapsed().as_millis();
        log::info!("Stage execution time: {}ms", times.elapsed().as_millis());
        log::info!("Total time elapsed: {}ms", total_millis);
        log::info!("============================");
        Ok(output)
    }
}
