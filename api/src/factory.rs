use compress_utils::context::Context;
use compress_utils::general_utils::DeviceEnum;
use std::sync::Arc;
use wgpu::Adapter;
use wgpu_compress_32_batched::decompressor::BatchedGPUDecompressor;
use wgpu_compress_32_batched::ChimpCompressorBatched;
use wgpu_compress_32_n_batched::chimpn::ChimpNGPUBatched;
use wgpu_compress_32_n_batched::decompressor::BatchedGPUNDecompressor;
use wgpu_compress_64_batched::decompressor::ChimpDecompressorBatched64;
use wgpu_compress_64_batched::ChimpCompressorBatched64;
use wgpu_compress_64_n_batched::chimpn::ChimpN64GPUBatched;
use wgpu_compress_64_n_batched::decompressor::GPUDecompressorBatchedN64;
use wgpu_types::{Backends, Limits};

// --- State Machine States ---
#[derive(Debug)]
pub struct NoAdapter;
#[derive(Debug)]
pub struct HasAdapter(pub Adapter);

#[derive(Debug)]
pub struct NoBufferSize;
#[derive(Debug)]
pub struct HasBufferSize(pub u32);

#[derive(Debug)]
pub struct NoN;
#[derive(Debug)]
pub struct HasN(pub u32);
// --- Builder Factory ---
#[derive(Debug)]
pub struct CompressorBuilder<A, B, C> {
    adapter: A,
    buffer_size: B,
    n: C,
}

impl CompressorBuilder<NoAdapter, NoBufferSize, NoN> {
    pub fn new() -> Self {
        CompressorBuilder {
            adapter: NoAdapter,
            buffer_size: NoBufferSize,
            n: NoN,
        }
    }
}

impl<B, C> CompressorBuilder<NoAdapter, B, C> {
    pub async fn with_default_adapter(self) -> CompressorBuilder<HasAdapter, B, C> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        CompressorBuilder {
            adapter: HasAdapter(adapter),
            buffer_size: self.buffer_size,
            n: self.n,
        }
    }

    pub fn with_adapter(self, adapter: Adapter) -> CompressorBuilder<HasAdapter, B, C> {
        CompressorBuilder {
            adapter: HasAdapter(adapter),
            buffer_size: self.buffer_size,
            n: self.n,
        }
    }
    pub async fn with_adapter_name(
        self,
        adapter: &impl Into<String>,
    ) -> Result<CompressorBuilder<HasAdapter, B, C>, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .enumerate_adapters(Backends::VULKAN)
            .into_iter()
            .find(|a| a.get_name() == adapter.into());

        if let None = adapter {
            return Err("Failed to find an appropriate adapter".to_string());
        }
        Ok(CompressorBuilder {
            adapter: HasAdapter(adapter),
            buffer_size: self.buffer_size,
            n: self.n,
        })
    }
}

impl<A, C> CompressorBuilder<A, NoBufferSize, C> {
    pub fn with_buffer_size(
        self,
        size: u32,
    ) -> Result<CompressorBuilder<A, HasBufferSize, C>, String> {
        if size % 256 != 0 {
            return Err("buffer_size must be a multiple of 256".to_string());
        }
        Ok(CompressorBuilder {
            adapter: self.adapter,
            buffer_size: HasBufferSize(size),
            n: self.n,
        })
    }
}
impl<A, B> CompressorBuilder<A, B, NoN> {
    pub fn with_n(self, n: u32) -> CompressorBuilder<A, B, HasN> {
        CompressorBuilder {
            adapter: self.adapter,
            buffer_size: self.buffer_size,
            n: HasN(n),
        }
    }
}

async fn create_context(adapter: Adapter) -> Arc<Context> {
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64,
            required_limits: Limits {
                max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
                max_buffer_size: adapter.limits().max_buffer_size,
                ..Limits::downlevel_defaults()
            },
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu_types::Trace::Off,
        })
        .await
        .expect("Failed to request a device");

    Arc::new(Context::new(device, queue, adapter))
}

// --- Final Build State ---
impl CompressorBuilder<HasAdapter, HasBufferSize, NoN> {
    pub async fn build32(self) -> (ChimpCompressorBatched, BatchedGPUDecompressor) {
        let adapter = self.adapter.0;
        let buffer_size = self.buffer_size.0;

        // Set buffer size in environment
        unsafe {
            std::env::set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
        }

        let context = create_context(adapter).await;

        let compressor = ChimpCompressorBatched::new(false, context.clone(), DeviceEnum::GPU);
        let decompressor = BatchedGPUDecompressor::new(context.clone());

        (compressor, decompressor)
    }

    pub async fn build64(self) -> (ChimpCompressorBatched64, ChimpDecompressorBatched64) {
        let adapter = self.adapter.0;
        let buffer_size = self.buffer_size.0;

        // Set buffer size in environment
        unsafe {
            std::env::set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
        }

        let context = create_context(adapter).await;

        let compressor = ChimpCompressorBatched64::default();
        let decompressor = ChimpDecompressorBatched64::new(context.clone());

        (compressor, decompressor)
    }
}

impl CompressorBuilder<HasAdapter, HasBufferSize, HasN> {
    pub async fn build32n(self) -> (ChimpNGPUBatched, BatchedGPUNDecompressor) {
        let adapter = self.adapter.0;
        let buffer_size = self.buffer_size.0;

        // Set buffer size in environment
        unsafe {
            std::env::set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
        }

        let context = create_context(adapter).await;

        let compressor = ChimpNGPUBatched::new(context.clone(), &*self.n);
        let decompressor = BatchedGPUNDecompressor::new(context.clone(), *self.n);

        (compressor, decompressor)
    }

    pub async fn build64n(self) -> (ChimpN64GPUBatched, GPUDecompressorBatchedN64) {
        let adapter = self.adapter.0;
        let buffer_size = self.buffer_size.0;

        // Set buffer size in environment
        unsafe {
            std::env::set_var("CHIMP_BUFFER_SIZE", buffer_size.to_string());
        }

        let context = create_context(adapter).await;

        let compressor = ChimpN64GPUBatched::new(context.clone(), &*self.n);
        let decompressor = GPUDecompressorBatchedN64::new(context.clone(), *self.n);

        (compressor, decompressor)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_size_validation() {
        let builder = CompressorBuilder::new();
        let res = builder.with_buffer_size(100);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), "buffer_size must be a multiple of 256");

        let builder = CompressorBuilder::new();
        let res = builder.with_buffer_size(256);
        assert!(res.is_ok());

        let builder = CompressorBuilder::new();
        let res = builder.with_buffer_size(512);
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_state_machine_logic() {
        // This test only verifies that the types compile and follow the state machine
        // We can't easily run full build tests without a GPU in CI environment sometimes,
        // but we can test the transition to HasBufferSize.

        let builder = CompressorBuilder::new();
        // builder.build32(); // Should NOT compile (and it doesn't because the method is not in scope)

        let builder_with_size = builder.with_buffer_size(256).unwrap();
        // builder_with_size.build32(); // Should NOT compile (missing adapter)

        // Final transition requires an adapter.
        // let builder_final = builder_with_size.with_default_adapter().await;
        // builder_final.build32().await; // This would work if we had an adapter.
    }
}
