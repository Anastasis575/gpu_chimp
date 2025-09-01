use crate::compute_s_shader::ComputeS;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::{ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::S;
use compress_utils::wgpu_utils::RunBuffers;
use compress_utils::{step, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::fs;
use std::iter::zip;
use std::ops::Div;
use std::sync::Arc;

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
    ) -> S {
        let n = self.n as u32;
        let full_size = ChimpBufferInfo::get().buffer_size() as u32;
        let v_u32: u32 = bytemuck::cast(v);
        let mut min_id = 1u32;
        let mut min_bit_count = n + 1u32;

        let mut new_id = id + 1u32;

        let start_of = new_id - (new_id % full_size);
        for index in 1u32..=n {
            let pre_start = (new_id < start_of + index);
            let actual_index = if (pre_start) {
                (start_of)
            } else {
                (new_id - index)
            };
            let v_prev = input_array[actual_index as usize];
            let mut v_prev_u32: u32 = bytemuck::cast(v_prev);
            let mut v_xorred = v_prev_u32 ^ v_u32;
            let mut leading =
                ((id % workgoup_size) != 0) as i32 * (v_xorred.leading_zeros() as i32);
            let mut trailing = (v_xorred.trailing_zeros() as i32);
            let mut equal = ((v_xorred == 0) as u32);
            let not_equal = ((v_xorred != 0) as u32);

            let center_bits = if equal == 0 {
                32u32 - ((leading) as u32) - ((trailing) as u32)
            } else {
                0
            };
            let case_1_bit_count = 2u32;
            let case_2_bit_count = 2u32 + 5u32 + 5u32 + center_bits;
            let case_3_and4_bit_count: u32 = 2u32 + 2u32 + 32u32 - ((leading) as u32); //The average of the required bit

            let trail_gt_6 = ((trailing > 6) as u32);
            let trail_le_6 = ((trailing <= 6) as u32);

            let mut final_bit_count = equal * case_1_bit_count
                + (trail_gt_6 * not_equal) * case_2_bit_count
                + (trail_le_6) * case_3_and4_bit_count;

            let is_min = final_bit_count < min_bit_count;
            min_bit_count = if (is_min) {
                final_bit_count
            } else {
                min_bit_count
            };
            min_id = if is_min { index } else { min_id };
        }
        let v_prev = input_array[(new_id - min_id) as usize];
        let mut v_prev_u32: u32 = bytemuck::cast(v_prev);
        let mut i = v_prev_u32 ^ v_u32;
        let mut leading = (id % workgoup_size != 0) as i32 * (i.leading_zeros() as i32);
        let mut trailing = (i.trailing_zeros() as i32);
        let mut equal = ((i == 0) as u32);
        let not_equal = ((i == 1) as u32);

        //   let mut leading_rounded:i32=((leading<8) as i32)*0;
        //   leading_rounded+=((leading>=8&&leading<12) as i32)*8;
        //   leading_rounded+=((leading>=12&&leading<16) as i32)*12;
        //   leading_rounded+=((leading>=16&&leading<18) as i32)*16;
        //   leading_rounded+=((leading>=18&&leading<20) as i32)*18;
        //   leading_rounded+=((leading>=20&&leading<22) as i32)*20;
        //   leading_rounded+=((leading>=20&&leading<24) as i32)*22;
        //   leading_rounded+=((leading>=24) as i32)*24;
        previous_index_array[new_id as usize] = min_id;
        return S {
            head: leading,
            tail: trailing,
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

        let mut padded_values = Vec::from(values);
        padded_values.push(0f32);
        let mut s_array = vec![S::default(); bytes];
        let mut previous_index = vec![0u32; bytes];
        let chunks = ChimpBufferInfo::get().chunks();
        for workgroup in (0..workgroup_count) {
            for invocation in 0..256 {
                for i in (0..chunks) {
                    let index: u32 = ((workgroup) * 256 * chunks + invocation + i * 256) as u32;
                    s_array[(index + 1) as usize] = self.calculate_s(
                        &padded_values,
                        &mut previous_index,
                        (chunks * 256) as u32,
                        index,
                        padded_values[(index + 1) as usize],
                    );
                }
            }
        }

        let input_buffer = BufferWrapper::storage_with_content(
            self.context.device(),
            bytemuck::cast_slice(&padded_values),
            WgpuGroupId::new(0, 0),
            Some("Input buffer"),
        );
        buffers.set_input_buffer(input_buffer);
        let s_buffer = BufferWrapper::storage_with_content(
            self.context.device(),
            bytemuck::cast_slice(&s_array),
            WgpuGroupId::new(0, 1),
            Some("Input buffer"),
        );
        buffers.set_s_buffer(s_buffer);
        let previous_buffer = BufferWrapper::storage_with_content(
            self.context.device(),
            bytemuck::cast_slice(&previous_index),
            WgpuGroupId::new(0, 2),
            Some("Input buffer"),
        );
        buffers.set_previous_index_buffer(previous_buffer);
        step!(Step::ComputeS, {
            let comb = zip(s_array, previous_index);
            comb.into_iter()
                .map(|(s, prev)| format!("{},{}\n", s, prev))
        });
        Ok(())
    }
}
