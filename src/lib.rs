use anyhow::{anyhow, Ok, Result};
use itertools::Itertools;
use pollster::FutureExt;
use wgpu::{Adapter, Device, Queue};
pub struct Context {
    device: Device,
    queue: Queue,
}


struct WgpuUtils{}

impl WgpuUtils{
    pub fn create_shader_module(device:&Device,shader_file:&String)->Result<wgpu::ShaderModule>{
        let cs_module=device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Compute_s"), source:wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("compute_s.wgsl")))  })
        Ok(cs_module)
    }
}


impl Context {
    pub fn new(device: Device, queue: Queue) -> Self {
        Self { device, queue }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    pub fn device_mut(&mut self) -> &mut Device {
        &mut self.device
    }
    pub fn queue_mut(&mut self) -> &mut Queue {
        &mut self.queue
    }

    pub async fn initialize_default_adapter() -> Result<Self> {
        Self::_initialize(None).await
    }
    pub async fn initialize_with_adapter(device: String) -> Result<Self> {
        Self::_initialize(Some(device)).await
    }

    async fn _initialize(device_name: Option<String>) -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter_list: Vec<Adapter>;
        let adapter = if let Some(device) = device_name {
            adapter_list = instance.enumerate_adapters(wgpu::Backends::all());
            let adapter = adapter_list
                .iter()
                .filter(|adapter| adapter.get_info().name.contains(device.as_str()))
                .find_or_first(|_| true)
                .ok_or(anyhow!("Not found"))
                .unwrap()
                .to_owned();
            adapter
        } else {
            &instance
                .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
                .await
                .ok_or(anyhow!("Not found"))?
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await?;

        Ok(Context::new(device, queue))
    }
}

struct ChimpCompressor {
    context: Context,
}
impl Default for ChimpCompressor {
    fn default() -> Self {
        let context = Context::initialize_default_adapter().block_on().unwrap();
        Self { context }
    }
}
impl ChimpCompressor {
    fn compute_s_module(&self)->Result<wgpu::ShaderModule>{
        // let cs_module=self.context().device().create_shader_module()
    }
    fn compute_s_shader(&self) -> Result<wgpu::ComputePipeline> {
        // Ok(wgpu::ComputePipeline::)
        
        
    }

    pub fn new(device: String) -> Self {
        let context = Context::initialize_with_adapter(device).block_on();
        Self { context }
    }
    pub fn chimpDecompress(values: &[u8]) -> Result<Vec<f32>> {
        //TODO
        Ok(Vec::new())
    }
    pub fn chimpCompress(values: &[f32]) -> Result<Vec<u8>> {
        let x = [0u8, 1u8];
        Ok(x.to_vec())
    }

    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn it_works() {
//         let result = add(2, 2);
//         assert_eq!(result, 4);
//     }
// }
