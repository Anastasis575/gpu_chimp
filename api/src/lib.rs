pub mod compressor32;
pub mod compressor64;
pub mod decompressor32;
pub mod decompressor64;
mod factory;


enum GPUMode {
    CPU,
    GPUIfAvailable(String),
    GPUMust(String),
}
#[derive(Default)]
struct BatchInfo {
    use_batches: bool,
    batch_numbers: u32,
}
impl BatchInfo {
    fn batches(self, batches: u32) -> BatchInfo {
        if batches == 0 {
            BatchInfo {
                use_batches: false,
                batch_numbers: 0,
            }
        } else {
            BatchInfo {
                use_batches: true,
                batch_numbers: batches,
            }
        }
    }
}

impl Default for GPUMode {
    fn default() -> Self {
        GPUMode::CPU
    }
}

// struct CompressorFactory {}
// 
// impl CompressorFactory {
//     fn builder_32() -> Builder32 {
//         Builder32::default()
//     }
// 
//     fn builder_64() -> Builder64 {
//         Builder64::default()
//     }
// }
// 
// #[derive(Default)]
// struct Builder32 {
//     driver: GPUMode,
//     batch: BatchInfo,
//     debug: bool,
// }
// impl Builder32 {
//     fn debug(self) -> Builder32 {
//         Builder32 {
//             debug: true,
//             ..self
//         }
//     }
//     fn no_debug(self) -> Builder32 {
//         Builder32 {
//             debug: false,
//             ..self
//         }
//     }
//     fn use_batches(self, batches: u32) -> Builder32 {
//         Builder32 {
//             batch: BatchInfo::default().batches(batches),
//             ..self
//         }
//     }
//     fn cpu(self) -> Builder32 {
//         Builder32 {
//             driver: CPU,
//             ..self
//         }
//     }
// 
//     fn gpu(self, adapter: impl Into<String>) -> Builder32 {
//         Builder32 {
//             driver: GPUMode::GPUIfAvailable(adapter.into()),
//             ..self
//         }
//     }
//     fn gpu_must(self, adapter: impl Into<String>) -> Builder32 {
//         Builder32 {
//             driver: GPUMode::GPUMust(adapter.into()),
//             ..self
//         }
//     }
// 
//     fn build(&self) -> Result<(ChimpCompressor32, ChimpDecompressor32), UtilError> {
//         match &self.driver {
//             CPU => Ok((
//                 cpu_compress::CPUCompressor::new(self.debug).into(),
//                 cpu_compress::CPUCompressor::new(self.debug).into(),
//             )),
//             GPUMode::GPUIfAvailable(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 };
//                 match context {
//                     Ok(context) => {
//                         let context: Arc<Context> = Arc::new(context);
//                         // if self.batch.use_batches {
//                         Ok((
//                             ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into(),
//                             BatchedGPUDecompressor::new(context.clone()).into(),
//                         ))
//                         // }
//                         // else {
//                         //     // (ChimpCompressor::new)
//                         // }
//                     }
//                     Err(_) => {
//                         info!("Couldn't instantiate adapter");
//                         Ok((
//                             cpu_compress::CPUCompressor::new(self.debug).into(),
//                             cpu_compress::CPUCompressor::new(self.debug).into(),
//                         ))
//                     }
//                 }
//             }
//             GPUMode::GPUMust(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 }?;
//                 let context = Arc::new(context);
//                 // if self.batch.use_batches {
//                 Ok((
//                     ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into(),
//                     BatchedGPUDecompressor::new(context.clone()).into(),
//                 ))
//                 // }
//                 // else {
//                 //     // (ChimpCompressor::new)
//                 // }
//             }
//         }
//     }
// 
//     fn build_compressor(&self) -> Result<ChimpCompressor32, UtilError> {
//         match &self.driver {
//             CPU => cpu_compress::CPUCompressor::new(self.debug).into(),
//             GPUMode::GPUIfAvailable(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 };
//                 match context {
//                     Ok(context) => {
//                         let context = Arc::new(context);
//                         // if self.batch.use_batches {
//                         ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into()
//                         // }
//                         // else {
//                         //     // (ChimpCompressor::new)
//                         // }
//                     }
//                     Err(_) => {
//                         info!("Couldn't instantiate adapter");
//                         cpu_compress::CPUCompressor::new(self.debug).into()
//                     }
//                 }
//             }
//             GPUMode::GPUMust(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 }?;
//                 let context = Arc::new(context);
//                 // if self.batch.use_batches {
//                 ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into()
//                 // }
//                 // else {
//                 //     // (ChimpCompressor::new)
//                 // }
//             }
//         }
//     }
//     fn build_decompressor(&self) -> Result<ChimpDecompressor32, UtilError> {
//         match &self.driver {
//             CPU => cpu_compress::CPUCompressor::new(self.debug).into(),
//             GPUMode::GPUIfAvailable(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 };
//                 match context {
//                     Ok(context) => {
//                         let context = Arc::new(context);
//                         // if self.batch.use_batches {
//                         BatchedGPUDecompressor::new(context.clone()).into()
//                         // }
//                         // else {
//                         //     // (ChimpCompressor::new)
//                         // }
//                     }
//                     Err(_) => {
//                         info!("Couldn't instantiate adapter");
//                         cpu_compress::CPUCompressor::new(self.debug).into()
//                     }
//                 }
//             }
//             GPUMode::GPUMust(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 }?;
//                 let context = Arc::new(context);
//                 // if self.batch.use_batches {
//                 BatchedGPUDecompressor::new(context.clone()).into()
//                 // }
//                 // else {
//                 //     // (ChimpCompressor::new)
//                 // }
//             }
//         }
//     }
// }
// 
// #[deprecated("needs 64 bit support")]
// #[derive(Default)]
// struct Builder64 {
//     driver: GPUMode,
//     batch: BatchInfo,
//     debug: bool,
// }
// 
// impl Builder64 {
//     fn debug(self) -> Builder64 {
//         Builder64 {
//             debug: true,
//             ..self
//         }
//     }
//     fn no_debug(self) -> Builder64 {
//         Builder64 {
//             debug: false,
//             ..self
//         }
//     }
//     fn use_batches(self, batches: u32) -> Builder64 {
//         Builder64 {
//             batch: BatchInfo::default().batches(batches),
//             ..self
//         }
//     }
//     fn cpu(self) -> Builder64 {
//         Builder64 {
//             driver: CPU,
//             ..self
//         }
//     }
// 
//     fn gpu(self, adapter: impl Into<String>) -> Builder64 {
//         Builder64 {
//             driver: GPUMode::GPUIfAvailable(adapter.into()),
//             ..self
//         }
//     }
//     fn gpu_must(self, adapter: impl Into<String>) -> Builder64 {
//         Builder64 {
//             driver: GPUMode::GPUMust(adapter.into()),
//             ..self
//         }
//     }
//     fn build(&self) -> Result<(ChimpCompressor64, ChimpDecompressor64), UtilError> {
//         match &self.driver {
//             CPU => Ok((
//                 cpu_compress::CPUCompressor::new(self.debug).into(),
//                 cpu_compress::CPUCompressor::new(self.debug).into(),
//             )),
//             GPUMode::GPUIfAvailable(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 };
//                 match context {
//                     Ok(context) => {
//                         let context: Arc<Context> = Arc::new(context);
//                         // if self.batch.use_batches {
//                         Ok((
//                             ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into(),
//                             BatchedGPUDecompressor::new(context.clone()).into(),
//                         ))
//                         // }
//                         // else {
//                         //     // (ChimpCompressor::new)
//                         // }
//                     }
//                     Err(_) => {
//                         info!("Couldn't instantiate adapter");
//                         Ok((
//                             cpu_compress::CPUCompressor::new(self.debug).into(),
//                             cpu_compress::CPUCompressor::new(self.debug).into(),
//                         ))
//                     }
//                 }
//             }
//             GPUMode::GPUMust(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 }?;
//                 let context = Arc::new(context);
//                 // if self.batch.use_batches {
//                 Ok((
//                     ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into(),
//                     BatchedGPUDecompressor::new(context.clone()).into(),
//                 ))
//                 // }
//                 // else {
//                 //     // (ChimpCompressor::new)
//                 // }
//             }
//         }
//     }
// 
//     fn build_compressor(&self) -> Result<ChimpCompressor64, UtilError> {
//         match &self.driver {
//             CPU => cpu_compress::CPUCompressor::new(self.debug).into(),
//             GPUMode::GPUIfAvailable(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 };
//                 match context {
//                     Ok(context) => {
//                         let context = Arc::new(context);
//                         // if self.batch.use_batches {
//                         ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into()
//                         // }
//                         // else {
//                         //     // (ChimpCompressor::new)
//                         // }
//                     }
//                     Err(_) => {
//                         info!("Couldn't instantiate adapter");
//                         cpu_compress::CPUCompressor::new(self.debug).into()
//                     }
//                 }
//             }
//             GPUMode::GPUMust(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 }?;
//                 let context = Arc::new(context);
//                 // if self.batch.use_batches {
//                 ChimpCompressorBatched::new(self.debug, context.clone(), GPU).into()
//                 // }
//                 // else {
//                 //     // (ChimpCompressor::new)
//                 // }
//             }
//         }
//     }
//     fn build_decompressor(&self) -> Result<ChimpDecompressor64, UtilError> {
//         match &self.driver {
//             CPU => cpu_compress::CPUCompressor::new(self.debug).into(),
//             GPUMode::GPUIfAvailable(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 };
//                 match context {
//                     Ok(context) => {
//                         let context = Arc::new(context);
//                         // if self.batch.use_batches {
//                         BatchedGPUDecompressor::new(context.clone()).into()
//                         // }
//                         // else {
//                         //     // (ChimpCompressor::new)
//                         // }
//                     }
//                     Err(_) => {
//                         info!("Couldn't instantiate adapter");
//                         cpu_compress::CPUCompressor::new(self.debug).into()
//                     }
//                 }
//             }
//             GPUMode::GPUMust(gpu) => {
//                 let context = if gpu.is_empty() {
//                     Context::initialize_default_adapter().block_on()
//                 } else {
//                     Context::initialize_with_adapter(gpu).block_on()
//                 }?;
//                 let context = Arc::new(context);
//                 // if self.batch.use_batches {
//                 BatchedGPUDecompressor::new(context.clone()).into()
//                 // }
//                 // else {
//                 //     // (ChimpCompressor::new)
//                 // }
//             }
//         }
//     }
// }

