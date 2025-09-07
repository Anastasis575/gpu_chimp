use crate::previous_indexes::PreviousIndexes;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::BufferWrapper;
use std::fs;
use std::ops::Div;
use std::sync::Arc;

pub struct PreviousIndexesNCPUImpl {
    pub(crate) context: Arc<Context>,
    pub(crate) n: usize,
}

impl PreviousIndexesNCPUImpl {
    pub fn new(context: Arc<Context>, n: usize) -> Self {
        Self { context, n }
    }

    pub fn context(&self) -> &Context {
        self.context.as_ref()
    }
    pub fn device(&self) -> &wgpu::Device {
        self.context.device()
    }
}

impl MaxGroupGnostic for PreviousIndexesNCPUImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}
#[async_trait]
impl PreviousIndexes for PreviousIndexesNCPUImpl {
    async fn calculate_previous_indexes(
        &self,
        values: &mut [f32],
        buffers: &mut RunBuffers,
        _skip_time: &mut u128,
    ) -> anyhow::Result<()> {
        let workgroup_count = self.get_max_number_of_groups(values.len());

        let bytes = values.len() + 1;

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);

        // let size = ChimpBufferInfo::get().buffer_size();
        let previous = vec![0u32; bytes];

        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();

        let mut writer = PreviousWriter {
            n: self.n as u32,
            input: padded_values,
            previous,
        };

        for i in 0..iterations {
            let offset = i * self.context.get_max_workgroup_size();
            for workgroup in 0..workgroup_count {
                let size = ChimpBufferInfo::get().buffer_size();
                writer.execute((offset + workgroup) * size, size)
            }
        }
        let input_storage_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(&writer.input),
            (0, 1),
            Some("Storage self.input Buffer"),
        );
        buffers.set_input_buffer(input_storage_buffer);
        let previous_index_buffer = BufferWrapper::storage_with_content(
            self.device(),
            bytemuck::cast_slice(&writer.previous),
            (0, 3),
            Some("Previous Index Buffer"),
        );

        buffers.set_previous_index_buffer(previous_index_buffer);

        if trace_steps().contains(&Step::PreviousIndexes) {
            let trace_path = Step::PreviousIndexes.get_trace_file();
            let mut trace_output = String::new();

            writer.previous.iter().for_each(|it| {
                let temp = format!("{}\n", it);
                trace_output.push_str(&temp);
            });

            fs::write(&trace_path, trace_output)?;
        }
        Ok(())
    }
}

struct PreviousWriter {
    input: Vec<f32>,
    previous: Vec<u32>,
    n: u32,
}
impl PreviousWriter {
    pub fn execute(&mut self, workgroup_start: usize, size: usize) {
        let mut indices = vec![0u32; 2usize.pow(self.n.ilog2() + 1)];
        let log2n = self.n.ilog2();
        let threshold = 2u32.pow(log2n + 1) - 1;
        let set_lsb: u32 = bytemuck::cast(threshold);
        let mut previous_index = 1usize;
        let threshold = 5u32 + (log2n);
        for step in 1usize + workgroup_start..size + workgroup_start {
            let value: u32 = bytemuck::cast(self.input[step]);
            let mut key = (value & set_lsb) as usize;
            let curr_index: usize = indices[key] as usize;
            if curr_index > 0 && (step - curr_index) < self.n as usize {
                let tempXor = value ^ bytemuck::cast::<f32, u32>(self.input[curr_index]);
                let trailingZeros = tempXor.trailing_zeros();

                if trailingZeros > threshold {
                    previous_index = step - curr_index;
                } else {
                    previous_index = 1usize;
                }
            } else {
                previous_index = 1usize;
            }
            indices[key] = step as u32;
            self.previous[step] = previous_index as u32;
        }
    }
}
