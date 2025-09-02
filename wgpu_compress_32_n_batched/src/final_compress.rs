use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::ChimpOutput;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{execute_compute_shader, wgpu_utils, BufferWrapper, WgpuGroupId};
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use std::time::Instant;
use wgpu_types::BufferAddress;

#[async_trait]
pub trait FinalCompress: MaxGroupGnostic {
    async fn final_compress(
        &self,
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> anyhow::Result<()>;
}

pub struct FinalCompressImpl {
    context: Arc<Context>,
    n: usize,
}

impl FinalCompressImpl {
    pub fn new(context: Arc<Context>, n: usize) -> Self {
        Self { context, n }
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref()
    }
}

impl MaxGroupGnostic for FinalCompressImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompress for FinalCompressImpl {
    async fn final_compress(
        &self,
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> anyhow::Result<()> {
        let size_of_output = size_of::<ChimpOutput>();
        let input_length = buffers.input_buffer().size() / size_of::<f32>();

        let output_buffer_size = (size_of_output * input_length) as BufferAddress;

        let workgroup_count = self.get_max_number_of_groups(input_length);

        let instant = Instant::now();
        let mut output_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size,
            WgpuGroupId::new(0, 2),
            Some("Storage Output Buffer"),
        );
        *skip_time += instant.elapsed().as_millis();
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
                .previous_index_buffer_mut()
                .with_binding(WgpuGroupId::new(0, 4));
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
            let log2n = format!("let log2n={}u;", self.n.ilog2());
            let temp = include_str!("shaders/chimp_compress.wgsl")
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@last_pass", &last_pass)
                .replace("//@log2n", &log2n)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &temp,
                vec![
                    buffers.s_buffer(),
                    buffers.input_buffer(),
                    buffers.previous_index_buffer(),
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
            let log2n = format!("let log2n={}u;", self.n.ilog2());
            let temp = include_str!("shaders/initialize_first_per_buffer.wgsl")
                .replace("//@workgroup_offset", &offset_decl)
                .replace("//@last_pass", &last_pass)
                .to_string();
            execute_compute_shader!(
                self.context(),
                &temp,
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
        if trace_steps().contains(&Step::Compress) {
            let output_staging_buffer = BufferWrapper::stage_with_size(
                self.context().device(),
                buffers.compressed_buffer().size() as BufferAddress,
                None,
            );
            let output = wgpu_utils::get_from_gpu::<ChimpOutput>(
                self.context(),
                buffers.compressed_buffer().buffer(),
                buffers.compressed_buffer().size() as BufferAddress,
                output_staging_buffer.buffer(),
            )
            .await?;

            let trace_path = Step::Compress.get_trace_file();
            let mut trace_output = String::new();

            output.iter().enumerate().for_each(|it| {
                trace_output.push_str(&format!("{}:{}\n", it.0, it.1));
            });

            fs::write(&trace_path, trace_output)?;
        }
        Ok(())
    }
}
