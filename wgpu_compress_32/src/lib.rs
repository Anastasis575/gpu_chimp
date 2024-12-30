use anyhow::{Ok, Result};
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::bit_utils::to_bit_vec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::Compressor;
use compress_utils::general_utils::{get_buffer_size, Padding};
use compress_utils::types::{ChimpOutput, S};
use compress_utils::{general_utils, time_it, wgpu_utils, BufferWrapper};
use general_utils::add_padding_to_fit_buffer_count;
use log::info;
use pollster::FutureExt;
use std::cmp::{max, min};
use std::ops::Div;
use std::process::Output;
use wgpu::{BufferAddress, Device, Queue};

///General methods for ChimpCompressor
impl ChimpCompressor {
    pub fn new(device: String, debug: bool) -> Result<Self> {
        let context = Context::initialize_with_adapter(device).block_on()?;
        Ok(Self { context, debug })
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

    fn collect_to_bit_vec(&self, input: &mut [f32], output: &Vec<ChimpOutput>) -> Result<BitVec> {
        let mut output_vec = to_bit_vec(input[0].to_bits());
        for value in output {
            if value.bit_count() >= 32 {
                for i in (0..(value.bit_count() - 32)).rev() {
                    output_vec.push((value.content_x() >> i) % 2 == 1)
                }
            }
            for i in (0..min(value.bit_count(), 32)).rev() {
                output_vec.push((value.content_y() >> i) % 2 == 1)
            }
        }
        Ok(output_vec)
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

///Compression Specific method implementations
impl ChimpCompressor {
    pub async fn compute_s(&self, values: &mut [f32]) -> Result<Vec<S>> {
        // Create shader module and pipeline
        let workgroup_size = format!("@workgroup_size({})", get_buffer_size());
        let temp = include_str!("shaders/compute_s.wgsl")
            .replace("#@workgroup_size(1)#", &workgroup_size)
            .to_string();
        let compute_s_shader_module = wgpu_utils::create_shader_module(self.device(), &temp)?;

        //Calculating buffer sizes and workgroup counts

        let size_of_s = size_of::<S>();
        let bytes = values.len() + 1;
        info!("The size of the input values vec: {}", bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        info!("The S buffer size in bytes: {}", s_buffer_size);

        let workgroup_count = values.len().div(get_buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(padded_values.as_slice()),
            Some("Storage Input Buffer"),
        );
        let s_staging_buffer =
            BufferWrapper::stage_with_size(self.device(), s_buffer_size, Some("Staging S Buffer"));
        let s_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            s_buffer_size,
            Some("Storage S Buffer"),
        );

        let binding_group_layout = wgpu_utils::assign_bind_groups(
            self.device(),
            vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer],
        );

        let compute_s_pipeline = wgpu_utils::create_compute_shader_pipeline(
            self.device(),
            &compute_s_shader_module,
            &binding_group_layout,
            Some("Compute s pipeline"),
        )?;
        let binding_group = wgpu_utils::create_bind_group(
            self.context(),
            &binding_group_layout,
            vec![&s_storage_buffer, &input_storage_buffer, &s_staging_buffer],
        );

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
            s_pass.dispatch_workgroups(max(workgroup_count, 1) as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));

        let output = wgpu_utils::get_s_output::<S>(
            self.context(),
            s_storage_buffer.buffer(),
            s_buffer_size,
            s_staging_buffer.buffer(),
        )
        .await?;
        // for (i, s) in output.iter().enumerate() {
        //     println!("{}:{:?}", i, s);
        // }
        if self.debug {
            for (i, o) in output.iter().enumerate() {
                info!("{i}:{:?}", o);
            }
        }
        info!("Output result size: {}", output.len());
        Ok(output)
    }
    //noinspection ALL
    async fn final_compress(
        &self,
        input: &mut Vec<f32>,
        s_values: &mut Vec<S>,
        padding: usize,
    ) -> Result<Vec<ChimpOutput>> {
        let workgroup_size = format!("@workgroup_size({})", get_buffer_size());
        let temp = include_str!("shaders/chimp_compress.wgsl")
            .replace("#@workgroup_size(1)#", &workgroup_size)
            .to_string();
        let final_compress_module = wgpu_utils::create_shader_module(self.device(), &temp)?;
        let size_of_s = size_of::<S>();
        let size_of_output = size_of::<ChimpOutput>();
        let input_length = input.len();
        info!("The length of the input vec: {}", input_length);

        let s_buffer_size = (size_of_s * s_values.len()) as BufferAddress;
        info!("The S buffer size in bytes: {}", &s_buffer_size);

        let output_buffer_size = (size_of_output * s_values.len()) as BufferAddress;
        info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = input.len().div(get_buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);
        let output_staging_buffer = BufferWrapper::stage_with_size(
            self.device(),
            output_buffer_size,
            Some("Staging S Buffer"),
        );
        let output_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            output_buffer_size,
            Some("Storage Output Buffer"),
        );
        let s_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(s_values.as_slice()),
            Some("Storage S Buffer"),
        );
        input.push(0f32);
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(input.as_slice()),
            Some("Storage Input Buffer"),
        );

        let binding_group_layout = wgpu_utils::assign_bind_groups(
            self.device(),
            vec![
                &s_storage_buffer,
                &input_storage_buffer,
                &output_storage_buffer,
                &output_staging_buffer,
            ],
        );
        let improve_s_pipeline = wgpu_utils::create_compute_shader_pipeline(
            self.device(),
            &final_compress_module,
            &binding_group_layout,
            Some("Compress pipeline"),
        )?;
        let binding_group = wgpu_utils::create_bind_group(
            self.context(),
            &binding_group_layout,
            vec![
                &s_storage_buffer,
                &input_storage_buffer,
                &output_storage_buffer,
                &output_staging_buffer,
            ],
        );
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
            s_pass.dispatch_workgroups(max(workgroup_count, 1) as u32, 1, 1)
        }

        self.queue().submit(Some(s_encoder.finish()));

        let output = wgpu_utils::get_s_output::<ChimpOutput>(
            self.context(),
            output_storage_buffer.buffer(),
            output_buffer_size,
            output_staging_buffer.buffer(),
        )
        .await?;
        if self.debug {
            for (i, o) in output.iter().enumerate() {
                println!("{i}:{}", o);
            }
        }
        let length_without_padding = output.len() - padding - 1;
        Ok(output[..length_without_padding].to_vec())
    }
}

pub struct ChimpCompressor {
    context: Context,
    debug: bool,
}
impl Default for ChimpCompressor {
    fn default() -> Self {
        let context = Context::initialize_default_adapter().block_on().unwrap();
        Self {
            context,
            debug: false,
        }
    }
}

#[async_trait]
impl Compressor for ChimpCompressor {
    async fn compress(&self, initial_values: &mut Vec<f32>) -> Result<Vec<u8>> {
        let mut padding = Padding(0);
        let buffer_size = get_buffer_size();

        let mut values = initial_values.to_owned();
        values = add_padding_to_fit_buffer_count(values, buffer_size, &mut padding);

        let mut total_millis = 0;
        let mut s_values: Vec<S>;
        let chimp_vec: Vec<ChimpOutput>;
        let output_vec: BitVec;
        time_it!(
            {
                s_values = self.compute_s(&mut values).await?;
            },
            total_millis,
            "s computation stage"
        );
        time_it!(
            {
                chimp_vec = self
                    .final_compress(&mut values, &mut s_values, padding.0)
                    .await?;
            },
            total_millis,
            "final output stage".to_string()
        );
        time_it!(
            {
                output_vec = self.collect_to_bit_vec(&mut values, &chimp_vec)?;
            },
            total_millis,
            "Result collection".to_string()
        );

        Ok(output_vec.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use bit_vec::BitVec;
    use compress_utils::bit_utils::{ceil_log2, to_bit_vec, BitReadable, BitWritable};

    #[allow(clippy::identity_op)]
    #[test]
    fn test1() {
        let x = 6u32;
        assert_eq!(x >> 0, x);
    }
    #[test]
    fn test2() {
        let x = 6u32;
        let t = to_bit_vec(x);
        assert_eq!("00000000000000000000000000000110", t.to_string());
    }
    #[test]
    fn test3() {
        let x = 6u32;
        let mut bit_vec = BitVec::new();
        bit_vec.write_bits(x, ceil_log2(x) + 2);
        assert_eq!("00110", bit_vec.to_string());
    }
    #[test]
    fn test4() {
        // let x = 6u32;
        // let mut bit_vec =BitVec::new();
        assert_eq!(3, ceil_log2(6));
        assert_eq!(4, ceil_log2(12));
        assert_eq!(5, ceil_log2(31));
    }
    #[test]
    fn test5() {
        let x = 6u32;
        let mut bit_vec = BitVec::new();
        bit_vec.write_bits(x, 3);
        bit_vec.write_bits(x + 2, 4);
        let position = ceil_log2(x);
        let size = ceil_log2(x + 2);
        assert_eq!(8, bit_vec.reinterpret_i32(position as usize, size as usize));
    }
    #[test]
    fn test6() {
        assert_eq!(1 << 6, 2_i32.pow(6));
        assert_eq!(0xff01 % (1 << 8), 1);
    }
}
