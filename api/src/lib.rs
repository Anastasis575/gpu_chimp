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
pub mod api {
    pub use crate::factory::CompressorBuilder;
    use wgpu::Adapter;

    pub fn list_adapters() -> Vec<Adapter> {
        let instance = wgpu::Instance::default();
        let adapter_list: Vec<Adapter>;
        adapter_list = instance.enumerate_adapters(wgpu::Backends::all());
        adapter_list
    }
}
