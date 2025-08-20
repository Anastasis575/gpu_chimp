use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::ChimpOutput64;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, step, wgpu_utils, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait FinalCompress: MaxGroupGnostic {
    async fn final_compress(&self, buffers: &mut RunBuffers) -> anyhow::Result<()>;
}

pub struct FinalCompressImpl64 {
    context: Arc<Context>,
    // debug: bool,
}

impl FinalCompressImpl64 {
    pub fn new(context: Arc<Context>, _debug: bool) -> Self {
        Self {
            context,
            // debug
        }
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref()
    }
}

impl MaxGroupGnostic for FinalCompressImpl64 {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompress for FinalCompressImpl64 {
    async fn final_compress(&self, buffers: &mut RunBuffers) -> anyhow::Result<()> {
        let utils_64 = include_str!("shaders/64_utils.wgsl");
        let temp = include_str!("shaders/chimp_compress.wgsl")
            .replace("//#include(64_utils)", utils_64)
            .to_string();
        let size_of_output = size_of::<ChimpOutput64>();
        let input_length = buffers.input_buffer().size() / size_of::<f64>();

        let output_buffer_size = (size_of_output * input_length) as BufferAddress;

        let workgroup_count = self.get_max_number_of_groups(input_length);
        let mut output_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size,
            WgpuGroupId::new(0, 2),
            Some("Storage Output Buffer"),
        );
        {
            buffers.s_buffer_mut().with_binding(WgpuGroupId::new(0, 0));
        }
        {
            buffers
                .input_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 1));
        }
        {
            buffers
                .chunks_uniform_mut()
                .with_binding(WgpuGroupId::new(0, 3));
        }
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let last_pass = format!(
                "let last_pass={}u;",
                if i == iterations - 1 { 1 } else { 0 }
            );
            let utils_64 = include_str!("shaders/64_utils.wgsl");
            let temp = include_str!("shaders/chimp_compress.wgsl")
                .replace("//#include(64_utils)", utils_64)
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@last_pass", &last_pass)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &temp,
                vec![
                    buffers.s_buffer(),
                    buffers.input_buffer(),
                    &output_storage_buffer,
                    buffers.chunks_uniform(),
                ],
                if i == iterations - 1 {
                    last_size
                } else {
                    self.context.get_max_workgroup_size()
                },
                Some("compress pass")
            );
        }

        {
            buffers
                .chunks_uniform_mut()
                .with_binding(WgpuGroupId::new(0, 2));
        }
        {
            buffers
                .input_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 1));
        }

        let initialize_shadder = include_str!("shaders/initialize_first_per_buffer.wgsl");
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for i in 0..iterations {
            let offset_decl = format!(
                "let workgroup_offset={}u;",
                i * self.context.get_max_workgroup_size()
            );
            let last_pass = format!(
                "let last_pass={}u;",
                if i == iterations - 1 { 1 } else { 0 }
            );
            execute_compute_shader!(
                self.context(),
                &initialize_shadder
                    .replace("//@workgroup_offset", &offset_decl)
                    .replace("//#include(64_utils)", utils_64),
                vec![
                    output_storage_buffer.with_binding(WgpuGroupId::new(0, 0)),
                    buffers.input_buffer(),
                    buffers.chunks_uniform()
                ],
                if i == iterations - 1 {
                    last_size
                } else {
                    self.context.get_max_workgroup_size()
                },
                Some("initialize pass")
            );
        }
        buffers.set_compressed_buffer(output_storage_buffer);
        let mut final_output;

        step!(Step::Compress, {
            let output_staging = BufferWrapper::stage_with_size(
                self.context().device(),
                buffers.compressed_buffer().size() as BufferAddress,
                None,
            );
            final_output = wgpu_utils::get_from_gpu::<ChimpOutput64>(
                self.context(),
                buffers.compressed_buffer().buffer(),
                buffers.compressed_buffer().size() as BufferAddress,
                output_staging.buffer(),
            )
            .await?;
            final_output
                .iter()
                .enumerate()
                .map(|it| format!("{}:{}\n", it.0, it.1))
                .into_iter()
        });
        Ok(())
    }
}
