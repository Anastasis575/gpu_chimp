mod utils;

use std::cmp::min;
use std::fs;
use std::ops::Div;
use anyhow::{anyhow, Ok, Result};
use async_trait::async_trait;
use bit_vec::BitVec;
use itertools::Itertools;
use pollster::FutureExt;
use wgpu::{Adapter, BindingType, Buffer, BufferAddress, BufferUsages, Device, Queue, ShaderModule, ShaderStages};
use wgpu::naga::MathFunction;

pub struct Context {
    device: Device,
    queue: Queue,
}

#[repr(C)]
#[derive(Clone, Default, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct S {
    head: i32,
    tail: i32,
    equal: u32,
    pr_lead: u32,
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

pub struct CompressSchema<'a> {
    compressor: &'a dyn Compressor,
    decompressor: &'a dyn Decompressor,
}

#[async_trait]
pub trait Compressor {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>>;
}
#[async_trait]
pub trait Decompressor {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>>;
}

pub struct ChimpCompressor {
    context: Context,
}
impl Default for ChimpCompressor {
    fn default() -> Self {
        let context = Context::initialize_default_adapter().block_on().unwrap();
        Self { context }
    }
}
#[repr(C)]
#[derive(Clone, Default, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ChimpOutput {
    content: [u32; 2],
    bit_count: u32,
}

#[async_trait]
impl Compressor for ChimpCompressor {
    async fn compress(&self, values: &mut Vec<f32>) -> Result<Vec<u8>> {
        let times = std::time::Instant::now();
        let mut total_milis = 0;
        if !fs::exists("debug.makoto")? {
            log::info!("No debug.makoto found");
        }
        log::info!("Starting s computation stage");
        log::info!("============================");
        let mut s_values: Vec<S>;
        let output_vec: BitVec;
        {
            s_values = self.compute_s(values).await?;
        }
        log::info!("============================");
        log::info!("Finished s computation stage");
        total_milis += times.elapsed().as_millis();
        log::info!("Stage execution time: {}ms", times.elapsed().as_millis());
        log::info!("Total time elapsed: {}ms", total_milis);
        log::info!("============================");

        log::info!("Started s propagation stage");
        log::info!("============================");
        let times = std::time::Instant::now();
        {
            s_values = self.improve_s(&mut s_values).await?;
        }
        log::info!("============================");
        log::info!("Finished s propagation stage");
        total_milis += times.elapsed().as_millis();
        log::info!("Stage execution time: {}ms", times.elapsed().as_millis());
        log::info!("Total time elapsed: {}ms", total_milis);
        log::info!("============================");

        log::info!("Starting final output stage");
        log::info!("============================");
        let times = std::time::Instant::now();
        {
            output_vec = self.final_compress(values, &mut s_values).await?;
        }
        log::info!("============================");
        log::info!("Finished final output stage");
        total_milis += times.elapsed().as_millis();
        log::info!("Stage execution time: {}ms", times.elapsed().as_millis());
        log::info!("Total time elapsed: {}ms", total_milis);
        log::info!("============================");

        if fs::exists("debug.makoto")? {
            let mut file = String::new();
            for s in s_values {
                file += format!("{:?}", s).as_str();
            }
            fs::write("text.txt", file).expect("could not write file");
        }
        Ok(output_vec.to_bytes())
    }
}

impl ChimpCompressor {
    pub fn new(device: String) -> Result<Self> {
        let context = Context::initialize_with_adapter(device).block_on()?;
        Ok(Self { context })
    }


    pub async fn compute_s(&self, values: &mut Vec<f32>) -> Result<Vec<S>> {
        // Create shader module and pipeline
        let temp = include_str!("shaders/compute_s.wgsl").to_string();
        let compute_s_shader_module = utils::WgpuUtils::create_shader_module(self.device(), &temp)?;

        //Calculating buffer sizes and workgroup counts

        if values.len() % 64 != 0 {
            let count = values.len() % 64;
            for _i in 0..count {
                values.push(0f32);
            }
        }

        let size_of_s = size_of::<S>();
        let bytes = values.len();
        log::info!("The size of the input values vec: {}",bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        log::info!("The S buffer size in bytes: {}",s_buffer_size);


        let workgroup_count = values.len() % 64;
        log::info!("The wgpu workgroup size: {}",&workgroup_count);


        let input_storage_buffer = utils::BufferWrapper::storage_with_content(self.device(), bytemuck::cast_slice(values), Some("Storage Input Buffer"));
        let s_staging_buffer = utils::BufferWrapper::stage_with_size(self.device(), s_buffer_size, Some("Staging S Buffer"));
        let s_storage_buffer = utils::BufferWrapper::storage_with_size(self.device(), s_buffer_size, Some("Storage S Buffer"));


        let binding_group_layout = utils::WgpuUtils::assign_bind_groups(self.device(), vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer]);

        let compute_s_pipeline = utils::WgpuUtils::create_compute_shader_pipeline(
            self.device(),
            &compute_s_shader_module,
            &binding_group_layout,
            Some("Compute s pipeline"),
        )?;
        let binding_group = utils::WgpuUtils::create_bind_group(self.context(), &binding_group_layout, vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer]);

        let mut s_encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut s_pass = s_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("s_pass"),
                timestamp_writes: None,
            });
            s_pass.set_pipeline(&compute_s_pipeline);
            s_pass.set_bind_group(0, &binding_group, &[]);
            s_pass.dispatch_workgroups(workgroup_count as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));


        let output = utils::WgpuUtils::get_s_output::<S>(self.context(), s_storage_buffer.buffer(), s_buffer_size, s_staging_buffer.buffer()).await?;
        log::info!("Output result size: {}",output.len());
        Ok(output)
    }


    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    pub fn device(&self) -> &Device {
        self.context.device()
    }

    pub fn queue(&self) -> &Queue {
        self.context.queue()
    }

    async fn improve_s(&self, s_values: &mut Vec<S>) -> Result<Vec<S>> {
        // Create shader module and pipeline
        let temp = include_str!("shaders/propagate_s.wgsl").to_string();
        let compute_s_shader_module = utils::WgpuUtils::create_shader_module(self.device(), &temp)?;

        let size_of_s = size_of::<S>();
        let bytes = s_values.len();
        log::info!("The size of the input values vec: {}",bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        log::info!("The S buffer size in bytes: {}",s_buffer_size);

        let workgroup_count = s_values.len() % 64;
        log::info!("The wgpu workgroup size: {}",&workgroup_count);

        let s_staging_buffer = utils::BufferWrapper::stage_with_size(self.device(), s_buffer_size, Some("Staging S Buffer"));
        let s_storage_buffer = utils::BufferWrapper::storage_with_content(self.device(), bytemuck::cast_slice(s_values.as_slice()), Some("Storage S Buffer"));

        let binding_group_layout = utils::WgpuUtils::assign_bind_groups(self.device(), vec![&s_storage_buffer, &s_staging_buffer]);
        let improve_s_pipeline = utils::WgpuUtils::create_compute_shader_pipeline(
            self.device(),
            &compute_s_shader_module,
            &binding_group_layout,
            Some("Compute s pipeline"),
        )?;
        let binding_group = utils::WgpuUtils::create_bind_group(self.context(), &binding_group_layout, vec![&s_storage_buffer, &s_staging_buffer]);
        let mut s_encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut s_pass = s_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("s_pass"),
                timestamp_writes: None,
            });
            s_pass.set_pipeline(&improve_s_pipeline);
            s_pass.set_bind_group(0, &binding_group, &[]);
            s_pass.dispatch_workgroups(workgroup_count as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));

        let output = utils::WgpuUtils::get_s_output::<S>(self.context(), s_storage_buffer.buffer(), s_buffer_size, s_staging_buffer.buffer()).await?;
        log::info!("Improved Output result size: {}",output.len());
        Ok(output)
    }

    async fn final_compress(&self, input: &mut Vec<f32>, s_values: &mut Vec<S>) -> Result<BitVec> {
        let temp = include_str!("shaders/chimp_compress.wgsl").to_string();
        let final_compress_module = utils::WgpuUtils::create_shader_module(self.device(), &temp)?;
        let size_of_s = size_of::<S>();
        let size_of_ouput = size_of::<ChimpOutput>();
        let input_length = input.len();
        log::info!("The length of the input vec: {}",input_length);

        let s_buffer_size = (size_of_s * input_length) as BufferAddress;
        log::info!("The S buffer size in bytes: {}",&s_buffer_size);

        let ouput_buffer_size = (size_of_ouput * input_length) as BufferAddress;
        log::info!("The Ouput buffer size in bytes: {}",&ouput_buffer_size);

        let workgroup_count = input.len() % 64;
        log::info!("The wgpu workgroup size: {}",&workgroup_count);
        let ouput_staging_buffer = utils::BufferWrapper::stage_with_size(self.device(), ouput_buffer_size, Some("Staging S Buffer"));
        let ouput_storage_buffer = utils::BufferWrapper::storage_with_size(self.device(), ouput_buffer_size, Some("Storage Output Buffer"));
        let s_storage_buffer = utils::BufferWrapper::storage_with_content(self.device(), bytemuck::cast_slice(s_values.as_slice()), Some("Storage S Buffer"));
        let input_storage_buffer = utils::BufferWrapper::storage_with_content(self.device(), bytemuck::cast_slice(input.as_slice()), Some("Storage Input Buffer"));

        let binding_group_layout = utils::WgpuUtils::assign_bind_groups(self.device(), vec![&s_storage_buffer, &input_storage_buffer, &ouput_storage_buffer, &ouput_staging_buffer]);
        let improve_s_pipeline = utils::WgpuUtils::create_compute_shader_pipeline(
            self.device(),
            &final_compress_module,
            &binding_group_layout,
            Some("Compress pipeline"),
        )?;
        let binding_group = utils::WgpuUtils::create_bind_group(self.context(), &binding_group_layout, vec![&s_storage_buffer, &input_storage_buffer, &ouput_storage_buffer, &ouput_staging_buffer]);
        let mut s_encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut s_pass = s_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compress_pass"),
                timestamp_writes: None,
            });
            s_pass.set_pipeline(&improve_s_pipeline);
            s_pass.set_bind_group(0, &binding_group, &[]);
            s_pass.dispatch_workgroups(workgroup_count as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));

        let output = utils::WgpuUtils::get_s_output::<ChimpOutput>(self.context(), ouput_storage_buffer.buffer(), ouput_buffer_size, ouput_staging_buffer.buffer()).await?;
        let mut output_vec = BitVec::new();
        for value in output {
            if value.bit_count > 32 {
                for i in (0..(value.bit_count - 32)).rev() {
                    output_vec.push((2_u32.pow(i) & value.content[0]) == 1)
                }
            }
            for i in (0..min(value.bit_count, 32)).rev() {
                output_vec.push((2_u32.pow(i) & value.content[0]) == 1)
            }
        }
        Ok(output_vec)
    }
}


struct CPUCompressor {}

impl CPUCompressor {

    pub fn reinterpret_num(&self, bit_vec: &BitVec, index: usize, offset: usize) -> u32 {
        let mut output = 0u32;
        for index in index..index + offset {
            output <<= 1;
            output += bit_vec[index] as u32;
        }
        output
    }
    fn ceil_log2(number:u32)->u32{
        let n=number.ilog2();
        if 2u32.pow(n)<=number{
            n+1
        }else {
            n
        }
    }
    pub fn write_bits(&self, bit_vec: &mut BitVec, number: u32) {
        for i in 0..Self::ceil_log2(number){
            bit_vec.push(number&2u32.pow(i)!=0);
        }
    }
}
#[async_trait]
impl Decompressor for CPUCompressor {
    async fn decompress(&self, vec: &mut Vec<u8>) -> Result<Vec<f32>> {
        let input_vector = BitVec::from_bytes(vec);
        let mut i = 0;
        let mut first_num_u32: u32 = 0u32;
        for i in 0..32 {
            first_num_u32 <<= 1;
            first_num_u32 += input_vector[i] as u32;
        }
        let first_num = first_num_u32 as f32;
        let mut output = vec![first_num];
        let mut last_num: u32 = first_num as u32;
        let mut last_lead: u32 = first_num as u32;
        while i < input_vector.len() {
            if !input_vector[i] {
                if !input_vector[i + 1] {
                    output.push(last_num as f32);
                    last_lead = 32;
                    i += 2;
                } else {
                    let lead = self.reinterpret_num(&input_vector, i + 2, 3);
                    let center = self.reinterpret_num(&input_vector, i + 5, 6);
                    let tail = 32 - lead - center;

                    let xor_plus_tail = self.reinterpret_num(&input_vector, i + 11, center as usize);
                    let number = (xor_plus_tail << tail) ^ (last_num);
                    last_num = number;
                    last_lead = lead;
                    output.push(last_num as f32);
                    i += 11 + center as usize;
                }
            } else if !input_vector[i + 1] {
                let xorred = self.reinterpret_num(&input_vector, i + 2, 32 - last_lead as usize);
                let number = xorred ^ last_num;
                last_num = number;
                output.push(last_num as f32);
                i += 2 + 32 - last_lead as usize;
            } else {
                let lead = self.reinterpret_num(&input_vector, i + 2, 3);
                let xorred = self.reinterpret_num(&input_vector, i + 2, 32 - lead as usize);
                let number = xorred ^ last_num;
                last_num = number;
                last_lead = lead;
                output.push(last_num as f32);
                i += 2 + 32 - last_lead as usize;
            }
        }
        Ok(output)
    }
}

#[async_trait]
impl Compressor for CPUCompressor {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>> {
        let mut bitVec = BitVec::new();

        for i in 0..32 {
            bitVec.push(2u32.pow(i) & (vec[0] as u32) == 1);
        }
        let mut last_lead = 0;
        for i in 1..vec.len(){
            let xorred=(vec[i] as u32)^(vec[i-1] as u32);
            let lead=xorred.leading_zeros() as usize;
            let trail=xorred.trailing_zeros() as usize;
            if trail>6{
                bitVec.push(false);
                if xorred==0{
                    bitVec.push(false);
                }else{
                    bitVec.push(true);
                    self.write_bits(&mut bitVec, (lead % 3) as u32);
                    let center_bits=32-lead-trail;
                    self.write_bits(&mut bitVec, (center_bits% 6) as u32);
                    self.write_bits(&mut bitVec, (xorred>>trail)% (center_bits as u32));
                }
            }else{
                bitVec.push(true);
                if lead== last_lead {
                    bitVec.push(false);
                    self.write_bits(&mut bitVec, xorred% (32-lead as u32));
                }else{
                    bitVec.push(true);
                    self.write_bits(&mut bitVec, (lead % 3) as u32);
                    self.write_bits(&mut bitVec, xorred% (32-lead as u32));
                }
            }
        }
        Ok(bitVec.to_bytes())
    }
}