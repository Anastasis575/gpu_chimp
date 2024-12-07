use anyhow::Result;
use async_trait::async_trait;
use bit_vec::BitVec;
use compress_utils::context::Context;
use compress_utils::cpu_compress::Compressor;
use compress_utils::general_utils::{
    add_padding_to_fit_buffer_count, get_buffer_size, ThisIsStupid,
};
use compress_utils::types::{ChimpOutput, S};
use compress_utils::{time_it, wgpu_utils, BufferWrapper};
use log::info;
use pollster::FutureExt;
use std::cmp::max;
use std::ops::Div;
use wgpu_types::BufferAddress;

#[derive(Debug)]
pub struct ChimpCompressorBatched {
    debug: bool,
    context: Context,
}
impl Default for ChimpCompressorBatched {
    fn default() -> Self {
        Self {
            debug: false,
            context: Context::initialize_default_adapter().block_on().unwrap(),
        }
    }
}

#[async_trait]
impl Compressor for ChimpCompressorBatched {
    async fn compress(&self, vec: &mut Vec<f32>) -> Result<Vec<u8>> {
        let mut padding = ThisIsStupid(0);
        let buffer_size = get_buffer_size();
        let mut values = vec.to_owned();
        values = add_padding_to_fit_buffer_count(values, buffer_size, &mut padding);
        let mut total_millis = 0;
        let mut s_values: Vec<S>;
        let mut chimp_vec: Vec<ChimpOutput>;
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
            "final output stage"
        );
        time_it!(
            {
                output_vec = BitVec::from_bytes(self.finalize(&mut chimp_vec).await?.as_slice());
            },
            total_millis,
            "final Result collection"
        );

        Ok(output_vec.to_bytes())
    }
}

impl ChimpCompressorBatched {
    pub fn new(debug: bool, context: Context) -> Self {
        Self { debug, context }
    }
    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }
    pub fn queue(&self) -> &wgpu::Queue {
        self.context.queue()
    }
    pub fn context(&self) -> &Context {
        &self.context
    }

    async fn compute_s(&self, values: &mut [f32]) -> Result<Vec<S>> {
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
        // if self.debug {
        //     for (i, o) in output.iter().enumerate() {
        //         info!("{i}:{:?}", o);
        //     }
        // }
        info!("Output result size: {}", output.len());
        Ok(output)
    }

    async fn finalize(&self, chimp_output: &mut Vec<ChimpOutput>) -> Result<Vec<u8>> {
        let temp = include_str!("shaders/chimp_finalize_compress.wgsl").to_string();
        let final_compress_module = wgpu_utils::create_shader_module(self.device(), &temp)?;
        // let size_of_chimp = size_of::<ChimpOutput>();
        let size_of_out = size_of::<u32>();

        let buffer_size = chimp_output.len(); //get_buffer_size();

        let input_length = chimp_output.len();
        info!("The length of the input vec: {}", input_length);

        let output_buffer_size = (size_of_out * chimp_output.len()) as BufferAddress;
        info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = chimp_output.len().div(get_buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let out_stage_buffer = BufferWrapper::stage_with_size(
            self.device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let out_storage_buffer = BufferWrapper::storage_with_size(
            self.device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let in_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(chimp_output.as_slice()),
            Some("Staging Output Buffer"),
        );
        let size_uniform = BufferWrapper::uniform_with_content(
            self.device(),
            bytemuck::cast_slice(buffer_size.to_ne_bytes().as_slice()),
            Some("Size Uniform Buffer"),
        );

        let useful_byte_count_storage = BufferWrapper::storage_with_size(
            self.device(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            Some("Useful Storage Buffer"),
        );
        let useful_byte_count_staging = BufferWrapper::stage_with_size(
            self.device(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            Some("Useful Staging Buffer"),
        );

        let binding_group_layout = wgpu_utils::assign_bind_groups(
            self.device(),
            vec![
                &out_stage_buffer,
                &out_storage_buffer,
                &in_storage_buffer,
                &size_uniform,
                &useful_byte_count_storage,
                &useful_byte_count_staging,
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
                &out_stage_buffer,
                &out_storage_buffer,
                &in_storage_buffer,
                &size_uniform,
                &useful_byte_count_storage,
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

        let output = wgpu_utils::get_s_output::<u32>(
            self.context(),
            out_storage_buffer.buffer(),
            output_buffer_size,
            out_stage_buffer.buffer(),
        )
        .await?;
        let indexes = wgpu_utils::get_s_output::<u32>(
            self.context(),
            useful_byte_count_storage.buffer(),
            (workgroup_count * size_of::<u32>()) as BufferAddress,
            useful_byte_count_staging.buffer(),
        )
        .await?;
        let mut final_vec = Vec::<u8>::new();
        for (i, useful_byte_count) in indexes.iter().enumerate() {
            let start_index = i * buffer_size;
            let byte_count = *useful_byte_count as usize;
            // for num in &output[start_index..start_index + byte_count + 1] {
            //     println!("{}", num);
            // }
            final_vec.extend(
                output[start_index..start_index + byte_count + 1]
                    .iter()
                    .flat_map(|it| it.to_ne_bytes()),
            );
        }
        Ok(final_vec)
    }
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
        let mut c = ChimpOutput::default();
        c.set_content_y(bytemuck::cast(input[0]));
        c.set_bit_count(32);
        let mut final_output = vec![c];
        final_output.extend(output[..length_without_padding].to_vec());
        Ok(final_output)
    }

    pub fn debug(&self) -> bool {
        self.debug
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }
}

#[cfg(test)]
mod tests {
    use crate::ChimpCompressorBatched;
    use compress_utils::cpu_compress::{CPUCompressor, Compressor, Decompressor};
    use compress_utils::general_utils::check_for_debug_mode;
    use compress_utils::types::ChimpOutput;
    use itertools::Itertools;
    use pollster::FutureExt;
    use std::{env, fs};

    fn get_third(field: &str) -> Option<String> {
        field
            .split(",")
            .collect_vec()
            .get(2)
            .map(|it| it.to_string())
    }
    fn get_values() -> anyhow::Result<Vec<f32>> {
        let dir = env::current_dir()?;
        let file_path = dir.parent().unwrap().join("city_temperature.csv");
        let file_txt = fs::read_to_string(file_path)?;
        let values = file_txt
            .split("\n")
            .map(get_third)
            .filter(|p| p.is_some())
            .map(|s| s.unwrap().parse::<f32>().unwrap())
            .collect_vec()
            .to_vec();
        Ok(values)
    }
    #[test]
    fn test1() {
        env_logger::init();

        let mut values = get_values().expect("Could not read test values")[..256].to_vec();
        log::info!("Starting compression of {} values", values.len());
        let mut compressor = ChimpCompressorBatched::default();
        if check_for_debug_mode().expect("Could not read file system") {
            compressor.set_debug(true);
        }
        let mut compressed_values = compressor.compress(&mut values).block_on().unwrap();

        let mut decompressor = CPUCompressor::default();
        let decompressed_values = decompressor
            .decompress(&mut compressed_values)
            .block_on()
            .unwrap();
    }
}
