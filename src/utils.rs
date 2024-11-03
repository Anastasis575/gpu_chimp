use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferAddress, BufferUsages, Device};

pub enum BufferWrapper {
    StorageBuffer { buffer: Buffer, size: usize },
    StagingBuffer { buffer: Buffer, size: usize },
}

impl BufferWrapper {
    pub fn buffer(&self) -> &Buffer {
        match self {
            BufferWrapper::StorageBuffer {
                buffer,
                size: _size,
            } => buffer,
            BufferWrapper::StagingBuffer {
                buffer,
                size: _size,
            } => buffer,
        }
    }
    pub fn stage_with_content(device: &Device, contents: &[u8], label: Option<&str>) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        });
        BufferWrapper::StagingBuffer {
            buffer,
            size: size_of_val(contents) / size_of::<u8>(),
        }
    }
    pub fn stage_with_size(device: &Device, size: BufferAddress, label: Option<&str>) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        BufferWrapper::StagingBuffer {
            buffer,
            size: size as usize,
        }
    }
    pub fn storage_with_content(device: &Device, contents: &[u8], label: Option<&str>) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        BufferWrapper::StorageBuffer {
            buffer,
            size: size_of_val(contents) / size_of::<u8>(),
        }
    }
    pub fn storage_with_size(device: &Device, size: BufferAddress, label: Option<&str>) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        BufferWrapper::StorageBuffer {
            buffer,
            size: size as usize,
        }
    }
}

/// WGPU utility functions
pub(crate) mod wgpu_utils {
    use crate::utils::BufferWrapper;
    use crate::Context;
    use anyhow::Result;
    use bytemuck::Pod;
    use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, ShaderModule};
    use wgpu_types::{BindingType, BufferAddress, ShaderStages};

    pub fn create_shader_module(device: &Device, shader_content: &String) -> Result<ShaderModule> {
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_content)),
        });
        Ok(cs_module)
    }
    pub fn assign_bind_groups(device: &Device, bindings: Vec<&BufferWrapper>) -> BindGroupLayout {
        let mut count = 0;
        let mut binding_group_layout_entries = Vec::<wgpu::BindGroupLayoutEntry>::new();
        for binding in &bindings {
            match binding {
                BufferWrapper::StorageBuffer { .. } => {
                    binding_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                        binding: count,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                    count += 1;
                }
                BufferWrapper::StagingBuffer { .. } => {}
            }
        }
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: binding_group_layout_entries.as_slice(),
        })
    }

    pub fn create_compute_shader_pipeline(
        device: &Device,
        shader_module: &ShaderModule,
        binding_layout: &BindGroupLayout,
        label: Option<&str>,
    ) -> Result<wgpu::ComputePipeline> {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label,
            bind_group_layouts: &[binding_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label,
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(compute_pipeline)
    }

    pub async fn get_s_output<T: Pod>(
        context: &Context,
        storage_buffer: &Buffer,
        bytes: BufferAddress,
        output_buffer: &Buffer,
    ) -> Result<Vec<T>> {
        let mut command_encoder = context
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        command_encoder.copy_buffer_to_buffer(storage_buffer, 0, output_buffer, 0, bytes);
        context.queue().submit(Some(command_encoder.finish()));
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        context
            .device()
            .poll(wgpu::Maintain::wait())
            .panic_on_timeout();
        receiver.recv_async().await??;
        let output: Vec<T> =
            bytemuck::cast_slice(buffer_slice.get_mapped_range()[..].iter().as_slice()).to_vec();
        output_buffer.unmap();
        anyhow::Ok(output)
    }

    pub fn create_bind_group(
        context: &Context,
        bind_group_layout: &BindGroupLayout,
        buffers: Vec<&BufferWrapper>,
    ) -> BindGroup {
        let mut entries = Vec::<wgpu::BindGroupEntry>::new();
        let mut count = 0;
        for buffer_wrap in buffers {
            match buffer_wrap {
                BufferWrapper::StorageBuffer { buffer, .. } => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: count,
                        resource: buffer.as_entire_binding(),
                    });
                    count += 1;
                }
                BufferWrapper::StagingBuffer { .. } => {}
            }
        }
        context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("s_calculation_layout"),
                layout: bind_group_layout,
                entries: entries.as_slice(),
            })
    }
}

/// Bit vector utility functions and traits
pub mod bit_utils {
    use bit_vec::BitVec;

    pub fn to_bit_vec(num: u32) -> BitVec {
        let mut bit_vec = BitVec::new();
        for i in (0..32).rev() {
            bit_vec.push((num >> i) % 2 == 1);
        }
        bit_vec
    }
    pub fn ceil_log2(number: u32) -> u32 {
        assert!(number > 0, "{}", number);
        let n = number.ilog2();
        if 2u32.pow(n) <= number {
            n + 1
        } else {
            n
        }
    }

    pub trait BitWritable {
        fn write_bits(&mut self, number: u32, size: u32);
    }
    impl BitWritable for BitVec {
        fn write_bits(&mut self, number: u32, size: u32) {
            if number != 0 {
                for i in (0..size).rev() {
                    self.push(number & 2u32.pow(i) != 0);
                }
            }
        }
    }
    pub trait BitReadable {
        fn reinterpret_u32(&self, index: usize, offset: usize) -> u32;
        fn reinterpret_i32(&self, index: usize, offset: usize) -> i32;
    }
    impl BitReadable for BitVec {
        fn reinterpret_u32(&self, index: usize, offset: usize) -> u32 {
            let mut output = 0u32;
            for i in index..index + offset {
                output <<= 1;
                output += self[i] as u32;
            }
            output
        }
        fn reinterpret_i32(&self, index: usize, offset: usize) -> i32 {
            let mut output = 0i32;
            for i in index..index + offset {
                output <<= 1;
                output += self[i] as i32;
            }
            output
        }
    }
}

/// General Utility Functions
pub mod general_utils {
    use std::fs;
    use std::fs::OpenOptions;

    pub fn check_for_debug_mode() -> anyhow::Result<bool> {
        Ok(fs::exists("anastasis.debug")?)
    }

    pub fn open_file_for_append(file_name: &str) -> anyhow::Result<fs::File> {
        Ok(OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_name)?)
    }
}
