use itertools::Itertools;
use thiserror::Error;
use wgpu::{Adapter, Device, Queue, RequestDeviceError};

#[derive(Debug)]
pub struct Context {
    device: Device,
    queue: Queue,
    // adapter: Adapter,
}

#[derive(Error, Debug)]
pub enum UtilError {
    #[error("Could not initialize the gpu context")]
    Unintialized,
    #[error("Could not find Adapter with this name {0}")]
    UnknownAdapter(String),
    #[error("Could not request Adapter with name {name}")]
    UnbindableAdapter {
        name: String,
        source: RequestDeviceError,
    },
}

impl Context {
    pub fn new(device: Device, queue: Queue, adapter: Adapter) -> Self {
        Self {
            device,
            queue,
            // adapter,
        }
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

    pub async fn initialize_default_adapter() -> Result<Self, UtilError> {
        Self::_initialize(None).await
    }
    pub async fn initialize_with_adapter(device: String) -> Result<Self, UtilError> {
        Self::_initialize(Some(device)).await
    }

    async fn _initialize(device_name: Option<String>) -> Result<Self, UtilError> {
        let instance = wgpu::Instance::default();

        let adapter_list: Vec<Adapter>;
        let adapter = if let Some(device) = device_name {
            adapter_list = instance.enumerate_adapters(wgpu::Backends::all());
            adapter_list
                .iter()
                .filter(|adapter| adapter.get_info().name.contains(device.as_str()))
                .find_or_first(|_| true)
                .ok_or(UtilError::UnknownAdapter(device.to_string()))?
                .to_owned()
        } else {
            instance
                .request_adapter(&wgpu::RequestAdapterOptionsBase::default())
                .await
                .ok_or(UtilError::Unintialized)?
        };

        // let max_workgroup_x_size=adapter.get_info*/
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
            .await
            .map_err(|source| UtilError::UnbindableAdapter {
                name: adapter.get_info().name.to_string(),
                source,
            })?;

        Ok(Context::new(device, queue, adapter))
    }
}
