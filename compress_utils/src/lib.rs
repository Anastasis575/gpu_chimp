pub mod context;
pub mod cpu_compress;
pub mod types;

use crate::BufferWrapper::UnInitialized;
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferAddress, BufferUsages, Device};

#[derive(Debug)]
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
    UnInitialized,
}
impl Default for BufferWrapper {
    fn default() -> Self {
        UnInitialized {}
    }
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
    pub fn size(&self) -> usize {
        match self {
            BufferWrapper::StorageBuffer { size, .. } => *size,
            BufferWrapper::StagingBuffer { size, .. } => *size,
            BufferWrapper::Uniform { size, .. } => *size,
            BufferWrapper::UnInitialized => unreachable!(),
        }
    }
    pub fn with_binding(&mut self, id: WgpuGroupId) -> &mut Self {
        match self {
            BufferWrapper::StorageBuffer { binding, group, .. } => {
                *binding = id.binding;
                *group = id.group;
                self
            }
            BufferWrapper::StagingBuffer { .. } => self,
            BufferWrapper::Uniform { binding, group, .. } => {
                *group = id.group;
                *binding = id.binding;
                self
            }
            BufferWrapper::UnInitialized { .. } => self,
        }
    }

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
            BufferWrapper::UnInitialized => unreachable!(),
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
    use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, Label, ShaderModule};
    use wgpu_types::PollType::Wait;
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

    /// Utility function to create a wgsl compute shader module to use in our pipeline from code defined in [shader_content]  
    pub fn create_shader_module(device: &Device, shader_content: &str) -> Result<ShaderModule> {
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_content)),
        });
        Ok(cs_module)
    }

    /// Utility function to create a bind group layout based on defined wrappers
    ///     
    /// Bind groups define which
    pub fn assign_bind_groups(device: &Device, bindings: Vec<&BufferWrapper>) -> BindGroupLayout {
        let mut binding_group_layout_entries = Vec::<wgpu::BindGroupLayoutEntry>::new();
        for binding in bindings {
            match *binding {
                BufferWrapper::StorageBuffer { binding, .. } => {
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
                BufferWrapper::StagingBuffer { .. } => {}
                BufferWrapper::Uniform { binding, .. } => {
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
                BufferWrapper::UnInitialized => {}
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

    pub async fn get_from_gpu<T: Pod>(
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
        let _result = context.device().poll(Wait)?.wait_finished();
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
        label: Option<&'static str>,
    ) -> BindGroup {
        let mut entries = Vec::<wgpu::BindGroupEntry>::new();
        let mut count = 0;
        for buffer_wrap in buffers {
            match buffer_wrap {
                BufferWrapper::StorageBuffer {
                    buffer, binding, ..
                } => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    });
                    count += 1;
                }
                BufferWrapper::StagingBuffer { .. } => {}
                BufferWrapper::Uniform {
                    buffer, binding, ..
                } => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    });
                    count += 1;
                }
                BufferWrapper::UnInitialized => {}
            }
        }
        context
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: if let Some(it) = label {
                    Some(it)
                } else {
                    Label::None
                },
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

    pub trait ToBitVec: Copy {
        fn to_bit_vec(self) -> BitVec;
    }

    // impl<T: Copy> ToBitVec for &T {
    //     fn to_bit_vec(self) -> BitVec {
    //         let x:T=self.to_owned();
    //         x.to_bit_vec()
    //     }
    // }
    impl ToBitVec for u32 {
        fn to_bit_vec(self) -> BitVec {
            let mut bit_vec = BitVec::new();
            for i in (0..32).rev() {
                bit_vec.push((self >> i) % 2 == 1);
            }
            bit_vec
        }
    }
    impl ToBitVec for u64 {
        fn to_bit_vec(self) -> BitVec {
            let mut bit_vec = BitVec::new();
            for i in (0..64).rev() {
                bit_vec.push((self >> i) % 2 == 1);
            }
            bit_vec
        }
    }
    impl ToBitVec for u8 {
        fn to_bit_vec(self) -> BitVec {
            let mut bit_vec = BitVec::new();
            for i in (0..32).rev() {
                bit_vec.push((self >> i) % 2 == 1);
            }
            bit_vec
        }
    }

    pub fn to_bit_vec_no_padding(num: u32) -> BitVec {
        let mut bit_vec = BitVec::new();
        for i in (0..ceil_log2(num)).rev() {
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

    pub trait BitWritable<T> {
        fn write_bits(&mut self, number: T, size: u32);
    }
    impl BitWritable<u32> for BitVec {
        fn write_bits(&mut self, number: u32, size: u32) {
            if number != 0 {
                for i in (0..size).rev() {
                    self.push(number & 2u32.pow(i) != 0);
                }
            }
        }
    }
    impl BitWritable<u64> for BitVec {
        fn write_bits(&mut self, number: u64, size: u32) {
            if number != 0 {
                for i in (0..size).rev() {
                    self.push(number & 2u64.pow(i) != 0);
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
    use log::warn;
    use std::collections::HashSet;

    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::fs::OpenOptions;
    use std::ops::Div;
    use std::path::PathBuf;
    use std::str::FromStr;

    #[derive(Debug, Default, Serialize, Deserialize)]
    pub struct CompressResult(pub Vec<u8>, pub usize);
    impl CompressResult {
        pub fn compressed_value_ref(&self) -> &Vec<u8> {
            &self.0
        }

        pub fn compressed_value(self) -> Vec<u8> {
            self.0
        }
        pub fn compressed_value_mut(&mut self) -> &mut Vec<u8> {
            &mut self.0
        }
        pub fn metadata_size(&self) -> usize {
            self.1
        }
    }
    impl From<CompressResult> for Vec<u8> {
        fn from(value: CompressResult) -> Self {
            value.0
        }
    }

    impl From<Vec<u8>> for CompressResult {
        fn from(value: Vec<u8>) -> Self {
            Self(value, 0usize)
        }
    }
    pub trait MaxGroupGnostic {
        fn get_max_number_of_groups(&self, content_len: usize) -> usize;
    }

    ///A struct to be able to borrow an usize representing a padding value
    pub struct Padding(pub usize);

    /// Add 0's to the end of [values] to be able to seamlessly work with the shadder's size requirements
    pub fn add_padding_to_fit_buffer_count(
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
    #[macro_export]
    macro_rules! step {
        ($step:expr,$writer_body:block) => {
            if trace_steps().contains(&$step) {
                let trace_path = $step.get_trace_file();
                let mut trace_output = String::new();

                $writer_body.for_each(|it| trace_output.push_str(it.as_str()));

                fs::write(&trace_path, trace_output).expect("Could not write step file");
            }
        };
    }
    #[macro_export]
    macro_rules! time_it_compound {
        ($var:block,$total_millis:expr,$stage_name:expr,$stage_count:expr,$timeStruct:expr) => {
            //info!("Starting {} #{}",$stage_name,$stage_count);
            //info!("============================");

            let times = std::time::Instant::now();

            $var

            //info!("============================");
            //info!("Finished {} #{}",$stage_name,$stage_count);

            $total_millis += times.elapsed().as_millis();

            //info!("Stage execution time: {}ms", times.elapsed().as_millis());
            //info!("============================");

            *$timeStruct.times.entry($stage_name.to_string()).or_insert(0)+=times.elapsed().as_millis();
            $timeStruct.total_time +=times.elapsed().as_millis();
        }
    }
    #[macro_export]
    macro_rules! time_it {
        ($var:block,$total_millis:expr,$stage_name:expr,$logger:expr) => {
            //info!("Starting {}",$stage_name);
            //info!("============================");

            let times = std::time::Instant::now();

            $var

            //info!("============================");
            //info!("Finished {}",$stage_name );

            $total_millis += times.elapsed().as_millis();

            //info!("Stage execution time: {}ms", times.elapsed().as_millis());
            //info!("Total time elapsed: {}ms", $total_millis);

            //info!("============================");
            //info!("============================");

            $logger($total_millis)
        };
        ($var:block,$total_millis:expr,$stage_name:expr) => {
            //info!("Starting {}",$stage_name);
            //info!("============================");

            let times = std::time::Instant::now();

            $var

            //info!("============================");
            //info!("Finished {}",$stage_name );

            $total_millis += times.elapsed().as_millis();

            //info!("Stage execution time: {}ms", times.elapsed().as_millis());
            //info!("Total time elapsed: {}ms", $total_millis);

            //info!("============================");
            //info!("============================");

        }

    }
    /// Macro to execute a compute shader in a WGPU context.
    ///
    /// This macro simplifies the process of creating and executing a compute shader by encapsulating
    /// the creation of a shader module, pipeline, bind group, and command encoder for dispatching
    /// the compute work. It handles various aspects of WGPU utilities, such as binding resources and
    /// dispatching compute workloads.
    ///
    /// # Parameters
    ///
    /// - `$context`: The WGPU `compress_utils::context::Context` to use.
    ///
    /// - `$shader_source`: The string source code for the compute shader. It should be in WGSL format.
    ///
    /// - `$buffers`: A list of GPU buffer (wrapped in `compress_utils::BufferWrapper`) objects to be used as bind group bindings in the compute shader. This
    ///   will be passed to `wgpu_utils::assign_bind_groups` and `wgpu_utils::create_bind_group` to create
    ///   the bind group layout and the actual bind group for the shader.
    ///
    /// - `$dispatch_size`: The number of workgroups to dispatch in the compute shader. The macro ensures
    ///   that at least 1 workgroup is dispatched.
    ///
    /// # Usage
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use compress_utils::context::Context;
    /// use compress_utils::{execute_compute_shader, BufferWrapper, WgpuGroupId};
    /// async {
    ///     let mut context = Arc::new(Context::initialize_default_adapter().await.unwrap()); //  WGPU context setup here
    ///     let shader_source = include_str!("path_to_shader.wgsl");
    ///     let uniform_bytes=0u32.to_be_bytes();
    ///     let mut buffers = vec![BufferWrapper::uniform_with_content(context.device(),&uniform_bytes,WgpuGroupId::new(0,1),Some("uniform_buffer"))]; // Buffers to be used in the compute shader
    ///     let dispatch_size = 64; // Number of workgroups to dispatch
    ///
    ///     // execute_compute_shader!(&mut context, shader_source, &mut buffers, dispatch_size, Some("test"));
    /// }
    /// ```
    #[macro_export]
    macro_rules! execute_compute_shader {
        ($context:expr,$shader_source:expr,$buffers:expr,$dispatch_size:expr,$binding_label:expr) => {
            let compute_shader_module =
                wgpu_utils::create_shader_module($context.device(), $shader_source)?;

            let binding_group_layout = wgpu_utils::assign_bind_groups($context.device(), $buffers);

            let compute_s_pipeline = wgpu_utils::create_compute_shader_pipeline(
                $context.device(),
                &compute_shader_module,
                &binding_group_layout,
                Some("Compute s pipeline"),
            )?;

            let binding_group = wgpu_utils::create_bind_group(
                $context,
                &binding_group_layout,
                $buffers,
                $binding_label,
            );

            let mut s_encoder = $context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut s_pass = s_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("s_pass"),
                    timestamp_writes: None,
                });
                s_pass.set_pipeline(&compute_s_pipeline);
                s_pass.set_bind_group(0, &binding_group, &[]);
                s_pass.dispatch_workgroups(max($dispatch_size, 1) as u32, 1, 1)
            }

            $context.queue().submit(Some(s_encoder.finish()));
        };
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

    /// This function retrieves the buffer size for the application, factoring in environment-specific
    /// configurations and performing the necessary validations.
    ///
    /// # Details
    /// The function checks for an environment variable `CHIMP_BUFFER_SIZE` to determine the buffer size.
    /// If the variable is not set or contains an invalid value, a default size of 256 values is used.
    /// The function ensures that the buffer size meets the following conditions:
    /// 1. The buffer size must be greater than 0.
    /// 2. The buffer size must be a multiple of 256.
    ///
    /// If these conditions are not met, the function will `panic` with an appropriate error message.
    ///
    /// Once the buffer size is determined and validated, it updates the environment variable
    /// `CHIMP_BUFFER_SIZE` to reflect the finalized value.
    ///
    /// # Returns
    /// A `ChimpBufferInfo` struct is returned, containing:
    /// - The final buffer size.
    /// - The number of 256-byte chunks that fit into the buffer size (calculated as `final_buffer / 256`).
    ///
    /// # Environment Variable
    /// - `CHIMP_BUFFER_SIZE`: Used as an override to set a custom buffer size. If not set or invalid, the
    ///   default value of 64 bytes will be used.
    ///
    /// # Panics
    /// - Panics if the final buffer size is not greater than 0.
    /// - Panics if the final buffer size is not a multiple of 256.
    ///
    /// # Example
    /// ```rust
    /// use compress_utils::general_utils::get_buffer_size;
    /// let buffer_info = get_buffer_size();
    /// println!("Buffer Size: {}, Chunks: {}", buffer_info.buffer_size(), buffer_info.chunks());
    /// ```
    pub fn get_buffer_size() -> ChimpBufferInfo {
        let default_buffer = 256usize;
        let final_buffer = match std::env::var("CHIMP_BUFFER_SIZE") {
            Ok(buffer_str) => buffer_str.parse::<usize>().unwrap_or_else(|_| {
                warn!("Buffer size specified but not in usize format... defaulting to 256");
                default_buffer
            }),
            Err(_) => {
                warn!("No explicit buffer size used... defaulting to 256");
                default_buffer
            }
        };
        assert!(final_buffer > 0, "Buffer size must be greater than 0");
        assert_eq!(
            final_buffer % 256,
            0,
            "Buffer size must be a multiple of 256"
        );
        std::env::set_var("CHIMP_BUFFER_SIZE", final_buffer.to_string());
        ChimpBufferInfo(final_buffer, final_buffer / 256)
    }

    /// A struct that represents information about a dedicated buffer for Chimp data processing.
    ///
    /// The `ChimpBufferInfo` struct encapsulates two pieces of information:
    /// - The size of the buffer in bytes.
    /// - The number of elements stored in the buffer.
    ///
    /// # Fields
    /// - `0`: A `usize` representing the size of the buffer in bytes.
    /// - `1`: A `usize` representing the number of elements present in the buffer.
    ///
    /// # Examples
    /// ```rust
    /// use compress_utils::general_utils::ChimpBufferInfo;
    /// let buffer_info = ChimpBufferInfo(1024, 256);
    /// println!("Buffer size: {} bytes, Elements: {}", buffer_info.buffer_size(), buffer_info.chunks());
    /// ```
    pub struct ChimpBufferInfo(usize, usize);
    impl ChimpBufferInfo {
        pub fn get() -> Self {
            get_buffer_size()
        }
        pub fn buffer_size(&self) -> usize {
            self.0
        }
        pub fn chunks(&self) -> usize {
            self.1
        }
    }

    #[derive(Eq, PartialEq, Hash)]
    pub enum Step {
        ComputeS,
        Compress,
        CalculateIndexes,
        Finalize,
        Decompress,
    }

    #[derive(Debug)]
    pub enum DeviceEnum {
        GPU,
        CPU,
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
                Step::CalculateIndexes => {
                    fs::create_dir_all("./traces/calculate_indexes/").unwrap();
                    PathBuf::from(format!(
                        "./traces/calculate_indexes/trace_{}.log",
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
                "decompress" => Ok(Step::Decompress),
                "calculate_indexes" => Ok(Step::CalculateIndexes),
                _ => Err(anyhow::anyhow!("Unknown step")),
            }
        }
    }
    pub fn trace_steps() -> HashSet<Step> {
        let mut default_options = HashSet::new();
        if let Ok(trace_options) = std::env::var("CHIMP_TRACE") {
            trace_options
                .split(",")
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
