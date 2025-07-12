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
struct CompressorFactory {}

impl CompressorFactory {
    fn builder_32() -> Builder32 {
        Builder32::default()
    }

    fn builder_64() -> Builder64 {
        Builder64::default()
    }
}

#[derive(Default)]
struct Builder32 {
    driver: GPUMode,
    batch: BatchInfo,
    debug: bool,
}
impl Builder32 {
    fn debug(self) -> Builder32 {
        Builder32 {
            debug: true,
            ..self
        }
    }
    fn no_debug(self) -> Builder32 {
        Builder32 {
            debug: false,
            ..self
        }
    }
    fn use_batches(self, batches: u32) -> Builder32 {
        Builder32 {
            batch: BatchInfo::default().batches(batches),
            ..self
        }
    }
    fn cpu(self) -> Builder32 {
        Builder32 {
            driver: CPU,
            ..self
        }
    }

    fn gpu(self, adapter: impl Into<String>) -> Builder32 {
        Builder32 {
            driver: GPUMode::GPUIfAvailable(adapter.into()),
            ..self
        }
    }
    fn gpu_must(self, adapter: impl Into<String>) -> Builder32 {
        Builder32 {
            driver: GPUMode::GPUMust(adapter.into()),
            ..self
        }
    }

    // fn build<T: Compressor<f32>, K: Decompressor<f64>>(
    //     &self,
    // ) -> Result<Box<dyn Compressor>, UtilError> {
    //     match &self.driver {
    //         CPU => Ok((
    //             cpu_compress::CPUCompressor::new(self.debug) as T,
    //             cpu_compress::CPUCompressor::new(self.debug) as K,
    //         )),
    //         GPUMode::GPUIfAvailable(gpu) => {
    //             let context = if gpu.is_empty() {
    //                 Context::initialize_default_adapter().block_on()
    //             } else {
    //                 Context::initialize_with_adapter(gpu).block_on()
    //             };
    //             match context {
    //                 Ok(context) => {
    //                     let context = Arc::new(context);
    //                     // if self.batch.use_batches {
    //                     Ok((
    //                         ChimpCompressorBatched::new(self.debug, context.clone(), GPU),
    //                         BatchedGPUDecompressor::new(context.clone()),
    //                     ))
    //                     // }
    //                     // else {
    //                     //     // (ChimpCompressor::new)
    //                     // }
    //                 }
    //                 Err(_) => {
    //                     info!("Couldn't instantiate adapter");
    //                     Ok((
    //                         cpu_compress::CPUCompressor::new(self.debug),
    //                         cpu_compress::CPUCompressor::new(self.debug),
    //                     ))
    //                 }
    //             }
    //         }
    //         GPUMode::GPUMust(gpu) => {
    //             let context = if gpu.is_empty() {
    //                 Context::initialize_default_adapter().block_on()
    //             } else {
    //                 Context::initialize_with_adapter(gpu).block_on()
    //             }?;
    //             let context = Arc::new(context);
    //             // if self.batch.use_batches {
    //             Ok((
    //                 ChimpCompressorBatched::new(self.debug, context.clone(), GPU),
    //                 BatchedGPUDecompressor::new(context.clone()),
    //             ))
    //             // }
    //             // else {
    //             //     // (ChimpCompressor::new)
    //             // }
    //         }
    //     }
    // }

    fn build_compressor(&self) -> impl Compressor<f32> {
        match &self.driver {
            CPU => cpu_compress::CPUCompressor::new(self.debug),
            GPUMode::GPUIfAvailable(gpu) => {
                let context = if gpu.is_empty() {
                    Context::initialize_default_adapter().block_on()
                } else {
                    Context::initialize_with_adapter(gpu).block_on()
                };
                match context {
                    Ok(context) => {
                        let context = Arc::new(context);
                        // if self.batch.use_batches {
                        ChimpCompressorBatched::new(self.debug, context.clone(), GPU)
                        // }
                        // else {
                        //     // (ChimpCompressor::new)
                        // }
                    }
                    Err(_) => {
                        info!("Couldn't instantiate adapter");
                        cpu_compress::CPUCompressor::new(self.debug)
                    }
                };
                x
            }
            GPUMode::GPUMust(gpu) => {
                let context = if gpu.is_empty() {
                    Context::initialize_default_adapter().block_on()
                } else {
                    Context::initialize_with_adapter(gpu).block_on()
                }?;
                let context = Arc::new(context);
                // if self.batch.use_batches {
                ChimpCompressorBatched::new(self.debug, context.clone(), GPU)
                // }
                // else {
                //     // (ChimpCompressor::new)
                // }
            }
        }
    }
    fn build_decompressor(&self) -> impl Decompressor<f32> {
        match &self.driver {
            CPU => Ok(cpu_compress::CPUCompressor::new(self.debug)),
            GPUMode::GPUIfAvailable(gpu) => {
                let context = if gpu.is_empty() {
                    Context::initialize_default_adapter().block_on()
                } else {
                    Context::initialize_with_adapter(gpu).block_on()
                };
                match context {
                    Ok(context) => {
                        let context = Arc::new(context);
                        // if self.batch.use_batches {
                        Ok(BatchedGPUDecompressor::new(context.clone()))
                        // }
                        // else {
                        //     // (ChimpCompressor::new)
                        // }
                    }
                    Err(_) => {
                        info!("Couldn't instantiate adapter");
                        Ok(cpu_compress::CPUCompressor::new(self.debug))
                    }
                }
            }
            GPUMode::GPUMust(gpu) => {
                let context = if gpu.is_empty() {
                    Context::initialize_default_adapter().block_on()
                } else {
                    Context::initialize_with_adapter(gpu).block_on()
                }?;
                let context = Arc::new(context);
                // if self.batch.use_batches {
                Ok(BatchedGPUDecompressor::new(context.clone()))
                // }
                // else {
                //     // (ChimpCompressor::new)
                // }
            }
        }
    }
}

struct ErrorStruct<T> {
    data: T,
    error: Option<UtilError>,
}
impl<T> ErrorStruct<T> {
    fn from(value: T) -> ErrorStruct<T> {
        Self {
            data: value,
            error: None,
        }
    }
}

#[async_trait]
impl<T, K> Compressor<T> for ErrorStruct<K>
where
    K: Compressor<T>,
{
    async fn compress(&self, vec: &mut Vec<T>) -> Result<Vec<u8>, CompressionError> {
        self.data.compress(vec)
    }
}
#[derive(Default)]
struct Builder64 {}
