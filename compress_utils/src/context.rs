use anyhow::anyhow;
use itertools::Itertools;
use wgpu::{Adapter, Device, Queue};

pub struct Context {
    device: Device,
    queue: Queue,
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

    pub async fn initialize_default_adapter() -> anyhow::Result<Self> {
        Self::_initialize(None).await
    }
    pub async fn initialize_with_adapter(device: String) -> anyhow::Result<Self> {
        Self::_initialize(Some(device)).await
    }

    async fn _initialize(device_name: Option<String>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter_list: Vec<Adapter>;
        let adapter = if let Some(device) = device_name {
            adapter_list = instance.enumerate_adapters(wgpu::Backends::all());
            adapter_list
                .iter()
                .filter(|adapter| adapter.get_info().name.contains(device.as_str()))
                .find_or_first(|_| true)
                .ok_or(anyhow!("Not found"))?
                .to_owned()
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
                    required_features: wgpu::Features::default(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None, //PLEASE ENABLE THE TRACE FEATURE,I NEED THIS
            )
            .await?;

        anyhow::Ok(Context::new(device, queue))
    }
}
