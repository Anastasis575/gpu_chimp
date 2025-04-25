pub mod context;
pub mod cpu_compress;
pub mod types;

use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferAddress, BufferUsages, Device};

/// Buffer Wrapper enum to encapsulate the assignment
pub enum BufferWrapper {
    StorageBuffer {
        buffer: Buffer,
        size: usize,
        group: u32,
        binding: u32,
    },
    StagingBuffer {
        buffer: Buffer,
        size: usize,
    },
    Uniform {
        buffer: Buffer,
        size: usize,
        group: u32,
        binding: u32,
    },
}

pub struct WgpuGroupId {
    group: u32,
    binding: u32,
}
impl WgpuGroupId {
    pub fn new(group: u32, binding: u32) -> Self {
        WgpuGroupId { group, binding }
    }
}
impl BufferWrapper {
    ///Buffer getter
    pub fn buffer(&self) -> &Buffer {
        match self {
            BufferWrapper::StorageBuffer {
                buffer,
                size: _size,
                ..
            } => buffer,
            BufferWrapper::StagingBuffer {
                buffer,
                size: _size,
            } => buffer,
            BufferWrapper::Uniform {
                buffer,
                size: _size,
                ..
            } => buffer,
        }
    }
    ///Create a staging buffer with pre-existing-content defined in the bytes in [contents] with an optional [label]
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

    ///Create an empty staging buffer with [size] in bytes and an optional [label]
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

    ///Create a storage buffer with pre-existing-content defined in the bytes of [contents] with an optional [label]
    pub fn storage_with_content(
        device: &Device,
        contents: &[u8],
        group: WgpuGroupId,
        label: Option<&str>,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        BufferWrapper::StorageBuffer {
            buffer,
            size: size_of_val(contents) / size_of::<u8>(),
            group: group.group,
            binding: group.binding,
        }
    }

    ///Create an empty staging buffer with [size] in bytes and an optional [label]
    pub fn storage_with_size(
        device: &Device,
        size: BufferAddress,
        wgpu_group_id: WgpuGroupId,
        label: Option<&str>,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        BufferWrapper::StorageBuffer {
            buffer,
            size: size as usize,
            group: wgpu_group_id.group,
            binding: wgpu_group_id.binding,
        }
    }
    ///Create a uniform buffer with pre-existing content defined in the bytes of [content]  and an optional [label]
    pub fn uniform_with_content(
        device: &Device,
        contents: &[u8],
        wgpu_group_id: WgpuGroupId,
        label: Option<&str>,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        BufferWrapper::Uniform {
            buffer,
            size: size_of_val(contents) / size_of::<u8>(),
            group: wgpu_group_id.group,
            binding: wgpu_group_id.binding,
        }
    }
}

/// WGPU utility functions
pub mod wgpu_utils {
    use crate::context::Context;
    use crate::cpu_compress::CompressionError;
    use crate::BufferWrapper;
    use anyhow::Result;
    use bytemuck::Pod;
    use thiserror::Error;
    use wgpu::naga::ShaderStage::Compute;
    use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, ShaderModule};
    use wgpu_types::{BindingType, BufferAddress, ShaderStages};

    /// Utility error description
    #[derive(Error, Debug)]
    pub enum WgpuUtilsError {
        #[error(transparent)]
        InvalidShaderException(#[from] anyhow::Error),
    }

    impl From<WgpuUtilsError> for CompressionError {
        /// Error conversion Method
        fn from(value: WgpuUtilsError) -> Self {
            CompressionError::FromBaseAnyhowError(anyhow::Error::from(value))
        }
    }

    pub enum ShaderType {
        WGSL,
        GLSL,
    }

    /// Utility function to create a wgsl compute shader module to use in our pipeline from code defined in [shader_content]
    pub fn create_shader_module(
        device: &Device,
        shader_content: &str,
        stype: ShaderType,
    ) -> Result<ShaderModule> {
        let binding = vec![("shaderFloat64", "VK_TRUE"), ("shaderInt64", "VK_TRUE")];
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: match stype {
                ShaderType::WGSL => {
                    wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_content))
                }
                ShaderType::GLSL => wgpu::ShaderSource::Glsl {
                    shader: std::borrow::Cow::Borrowed(shader_content),
                    defines: binding.as_slice(),
                    stage: Compute,
                },
            },
        });
        Ok(cs_module)
    }

    /// Utility function to create a bind group layout based on defined wrappers
    ///
    /// Bind groups define which
    pub fn assign_bind_groups(device: &Device, bindings: Vec<&BufferWrapper>) -> BindGroupLayout {
        let mut binding_group_layout_entries = Vec::<wgpu::BindGroupLayoutEntry>::new();
        for binding in bindings {
            match binding {
                &BufferWrapper::StorageBuffer { binding, .. } => {
                    binding_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
                &BufferWrapper::StagingBuffer { .. } => {}
                &BufferWrapper::Uniform { binding, .. } => {
                    binding_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                        binding,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform {},
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    });
                }
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
        let _ = context.device().poll(wgpu::PollType::Wait)?.wait_finished();

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
                BufferWrapper::Uniform { buffer, .. } => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: count,
                        resource: buffer.as_entire_binding(),
                    });
                    count += 1;
                }
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
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum BitError {}

    pub fn to_bit_vec(num: u32) -> BitVec {
        let mut bit_vec = BitVec::new();
        for i in (0..32).rev() {
            bit_vec.push((num >> i) % 2 == 1);
        }
        bit_vec
    }
    pub fn to_bit_vec_no_padding(num: u32) -> BitVec {
        let mut bit_vec = BitVec::new();
        for i in (0..ceil_log2(num)).rev() {
            bit_vec.push((num >> i) % 2 == 1);
        }
        bit_vec
    }
    pub fn to_bit_vec_u8(num: u8) -> BitVec {
        let mut bit_vec = BitVec::new();
        for i in (0..8).rev() {
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
        fn reinterpret_u64(&self, index: usize, offset: usize) -> u64;
        fn reinterpret_i64(&self, index: usize, offset: usize) -> i64;
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
        fn reinterpret_u64(&self, index: usize, offset: usize) -> u64 {
            let mut output = 0u64;
            for i in index..index + offset {
                output <<= 1;
                output += self[i] as u64;
            }
            output
        }
        fn reinterpret_i64(&self, index: usize, offset: usize) -> i64 {
            let mut output = 0i64;
            for i in index..index + offset {
                output <<= 1;
                output += self[i] as i64;
            }
            output
        }
    }
}

/// General Utility Functions
pub mod general_utils {
    use log::warn;
    use std::collections::HashSet;

    use std::fs;
    use std::fs::OpenOptions;
    use std::ops::Div;
    use std::path::PathBuf;
    use std::str::FromStr;

    pub trait MaxGroupGnostic {
        fn get_max_number_of_groups(&self, content_len: usize) -> usize;
    }

    ///A struct to be able to borrow a usize representing a padding
    pub struct Padding(pub usize);

    /// Add 0s to the end of [values] to be able to seemlessly
    pub fn add_padding_to_fit_buffer_count_f32(
        mut values: Vec<f32>,
        buffer_size: usize,
        padding: &mut Padding,
    ) -> Vec<f32> {
        if values.len() % buffer_size != 0 {
            let count = (values.len().div(buffer_size) + 1) * buffer_size - values.len();
            padding.0 = count;
            for _i in 0..count {
                values.push(0f32);
            }
        }
        values
    }
    /// Add 0s to the end of [values] to be able to seemlessly
    pub fn add_padding_to_fit_buffer_count_f64(
        mut values: Vec<f64>,
        buffer_size: usize,
        padding: &mut Padding,
    ) -> Vec<f64> {
        if values.len() % buffer_size != 0 {
            let count = (values.len().div(buffer_size) + 1) * buffer_size - values.len();
            padding.0 = count;
            for _i in 0..count {
                values.push(0f64);
            }
        }
        values
    }

    #[macro_export]
    macro_rules! time_it_compound {
        ($var:block,$total_millis:expr,$stage_name:expr,$stage_count:expr,$timeStruct:expr) => {
            info!("Starting {} #{}",$stage_name,$stage_count);
            info!("============================");
            let times = std::time::Instant::now();
            $var
            info!("============================");
            info!("Finished {} #{}",$stage_name,$stage_count);
            $total_millis += times.elapsed().as_millis();
            info!("Stage execution time: {}ms", times.elapsed().as_millis());
            info!("============================");
            *$timeStruct.times.entry($stage_name.to_string()).or_insert(0)+=times.elapsed().as_millis();
            $timeStruct.total_time +=times.elapsed().as_millis();
        }
    }
    #[macro_export]
    macro_rules! time_it {
        ($var:block,$total_millis:expr,$stage_name:expr) => {
        info!("Starting {}",$stage_name);
        info!("============================");
        let times = std::time::Instant::now();
        $var
        info!("============================");
        info!("Finished {}",$stage_name );
        $total_millis += times.elapsed().as_millis();
        info!("Stage execution time: {}ms", times.elapsed().as_millis());
        info!("Total time elapsed: {}ms", $total_millis);
        info!("============================");
        info!("============================");
        }
    }

    pub fn check_for_debug_mode() -> anyhow::Result<bool> {
        let debug = false;
        let final_debug = if let Ok(buffer_str) = std::env::var("CHIMP_DEBUG") {
            if buffer_str == "true" {
                true
            } else {
                debug
            }
        } else {
            debug
        };
        Ok(final_debug)
    }

    pub fn get_buffer_size() -> usize {
        let default_buffer = 64usize;
        let final_buffer = match std::env::var("CHIMP_BUFFER_SIZE") {
            Ok(buffer_str) => buffer_str.parse::<usize>().unwrap_or_else(|_| {
                warn!("Buffer size specified but not in usize format... defaulting to 32");
                default_buffer
            }),
            Err(_) => {
                warn!("No explicit buffer size used... defaulting to 32");
                default_buffer
            }
        };
        std::env::set_var("CHIMP_BUFFER_SIZE", final_buffer.to_string());
        final_buffer
    }

    #[derive(Eq, PartialEq, Hash)]
    pub enum Step {
        ComputeS,
        Compress,
        Finalize,
        Decompress,
    }
    impl Step {
        pub fn get_trace_file(&self) -> PathBuf {
            match self {
                Step::ComputeS => {
                    fs::create_dir_all("./traces/compute_s/").unwrap();
                    PathBuf::from(format!(
                        "./traces/compute_s/trace_{}.log",
                        chrono::Local::now().to_utc()
                    ))
                }
                Step::Compress => {
                    fs::create_dir_all("./traces/compress/").unwrap();
                    PathBuf::from(format!(
                        "./traces/compress/trace_{}.log",
                        chrono::Local::now().to_utc()
                    ))
                }
                Step::Finalize => {
                    fs::create_dir_all("./traces/finalize/").unwrap();
                    PathBuf::from(format!(
                        "./traces/finalize/trace_{}.log",
                        chrono::Local::now().to_utc()
                    ))
                }
                Step::Decompress => {
                    fs::create_dir_all("./traces/decompress/").unwrap();
                    PathBuf::from(format!(
                        "./traces/decompress/trace_{}.log",
                        chrono::Local::now().to_utc()
                    ))
                }
            }
        }
    }
    impl FromStr for Step {
        type Err = anyhow::Error;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "compute_s" => Ok(Step::ComputeS),
                "compress" => Ok(Step::Compress),
                "finalize" => Ok(Step::Finalize),
                _ => Err(anyhow::anyhow!("Unknown step")),
            }
        }
    }
    pub fn trace_steps() -> HashSet<Step> {
        let mut default_options = HashSet::new();
        if let Ok(trace_options) = std::env::var("CHIMP_TRACE") {
            trace_options
                .split(";")
                .flat_map(|it| it.parse::<Step>())
                .for_each(|it| {
                    default_options.insert(it);
                });
        } else {
            warn!("No explicit trace options specified... defaulting to None");
        }
        default_options
    }

    pub fn open_file_for_append(file_name: &str) -> anyhow::Result<fs::File> {
        Ok(OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_name)?)
    }
}
