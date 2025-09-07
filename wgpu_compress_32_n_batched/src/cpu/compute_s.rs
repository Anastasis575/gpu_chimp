#![allow(unused_parens)]
#![allow(unused_variables)]

use crate::compute_s_shader::ComputeS;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::{ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::S;
use compress_utils::wgpu_utils::{get_from_gpu, RunBuffers};
use compress_utils::{step, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use wgpu_types::BufferAddress;

pub struct CPUBatchedNComputeSImpl {
    pub(crate) context: Arc<Context>,
    pub(crate) n: usize,
}

impl CPUBatchedNComputeSImpl {
    pub(crate) fn calculate_s(
        &self,
        input_array: &Vec<f32>,
        previous_index_array: &mut Vec<u32>,
        workgoup_size: u32,
        id: u32,
        v: f32,
        v_prev: f32,
    ) -> S {
        let n = self.n as u32;
        let full_size = ChimpBufferInfo::get().buffer_size() as u32;
        let v_u32: u32 = bytemuck::cast(v);
        let v_prev_u32: u32 = bytemuck::cast(v_prev);
        let i = v_prev_u32 ^ v_u32;

        let leading = if ((id % workgoup_size) != 0) {
            i.leading_zeros()
        } else {
            0
        };
        let trailing = i.trailing_zeros();
        let equal = (i == 0) as u32;
        return S {
            head: leading as i32,
            tail: trailing as i32,
            equal,
        };
    }
}

impl MaxGroupGnostic for CPUBatchedNComputeSImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl ComputeS for CPUBatchedNComputeSImpl {
    async fn compute_s(
        &self,
        values: &mut [f32],
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> anyhow::Result<()> {
        let workgroup_count = self.get_max_number_of_groups(values.len());
        //info!("The wgpu workgroup size: {}", &workgroup_count);

        let size_of_s = size_of::<S>();
        let bytes = values.len() + 1;

        let staging_buffer = BufferWrapper::stage_with_size(
            self.context.device(),
            buffers.previous_index_buffer().size() as BufferAddress,
            None,
        );
        let mut previous_index = get_from_gpu::<u32>(
            self.context.as_ref(),
            buffers.previous_index_buffer().buffer(),
            buffers.previous_index_buffer().size() as BufferAddress,
            staging_buffer.buffer(),
        )
        .await?;

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);
        let mut s_array = vec![S::default(); bytes];
        // let mut previous_index = vec![0u32; bytes];
        let chunks = ChimpBufferInfo::get().chunks();
        for workgroup in (0..workgroup_count) {
            for invocation in 0..256 {
                for i in (0..chunks) {
                    let index: u32 = ((workgroup) * 256 * chunks + invocation + i * 256) as u32;
                    let prev_id = (index + 1 - previous_index[(index + 1) as usize]) as usize;
                    s_array[(index + 1) as usize] = self.calculate_s(
                        &padded_values,
                        &mut previous_index,
                        (chunks * 256) as u32,
                        index,
                        padded_values[(index + 1) as usize],
                        padded_values[prev_id],
                    );
                }
            }
        }

        // let input_buffer = BufferWrapper::storage_with_content(
        //     self.context.device(),
        //     bytemuck::cast_slice(&padded_values),
        //     WgpuGroupId::new(0, 0),
        //     Some("Input buffer"),
        // );
        // buffers.set_input_buffer(input_buffer);
        let s_buffer = BufferWrapper::storage_with_content(
            self.context.device(),
            bytemuck::cast_slice(&s_array),
            WgpuGroupId::new(0, 0),
            Some("Input buffer"),
        );
        buffers.set_s_buffer(s_buffer);
        let chunk_buffer = BufferWrapper::uniform_with_content(
            self.context.device(),
            bytemuck::bytes_of(&ChimpBufferInfo::get().chunks()),
            WgpuGroupId::new(0, 3),
            Some("Input buffer"),
        );
        buffers.set_chunks(chunk_buffer);
        step!(Step::ComputeS, {
            s_array.into_iter().map(|s| format!("{}\n", s))
        });
        Ok(())
    }
}
