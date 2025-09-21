#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(unused_mut)]
#![allow(unused_mut)]

use crate::final_compress::FinalCompressN64;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::{ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::{ChimpOutput64, S};
use compress_utils::wgpu_utils::{get_from_gpu, RunBuffers};
use compress_utils::{step, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use wgpu_compress_64_batched::cpu::utils_64;
use wgpu_compress_64_batched::cpu::utils_64::{pseudo_u64_shift, vec2, vec_condition};
use wgpu_types::BufferAddress;

pub struct CPUBatchedNCompressImpl {
    pub(crate) context: Arc<Context>,
    pub(crate) n: usize,
}

impl CPUBatchedNCompressImpl {
    pub fn compress(&self, s: S, v: f64, s_prev: S, v_prev: f64, c: usize) -> ChimpOutput64 {
        let log2n = self.n.ilog2();
        //Conditions
        let mut trail_gt_6 = (s.tail > 6) as u32;
        let mut trail_le_6 = (s.tail <= 6) as u32;
        let mut not_equal = (s.equal != 1) as u32;
        let mut pr_lead = (s_prev.head) as u32;
        let mut pr_lead_eq_lead = (s.head == (pr_lead) as i32) as u32;
        let mut pr_lead_ne_lead = (s.head != (pr_lead) as i32) as u32;

        //Constants
        //0x1000 0000 0000 0000 0000 0000 0000 0000
        let mut first_bit_one: u32 = 0x80000000;

        //input
        let mut v_prev_u32: u64 = bytemuck::cast(v_prev);
        let mut v_u32: u64 = bytemuck::cast(v);
        let mut xorred: u64 = v_prev_u32 ^ v_u32;

        let mut center_bits = if (s.equal != 1) {
            (64 - s.head - s.tail) as u32
        } else {
            64
        };

        //case 1:  xor_value=0
        let mut case_1: vec2<u64> = vec2::<u64>(0, c as u64);
        let mut case_1_bit_count: u32 = 2;

        //    let mut head_representation=(s.head>=8&&s.head<12)*1+u32(s.head>=12&&s.head<16)*2+u32(s.head>=16&&s.head<18)*3+u32(s.head>=18&&s.head<20)*4+u32(s.head>=20&&s.head<22)*5+u32(s.head>=22&&s.head<24)*6+u32(s.head>=24) as u32*7;

        // case 2: tail>6 && xor_value!=0(!equal)
        let mut case_2: vec2<u64> = vec2::<u64>(0u64, 1u64); //code:01 bit_count=2
        case_2 = pseudo_u64_shift(case_2, log2n);
        case_2.1 += utils_64::extract_bits(c as u64, 0u32, log2n);
        case_2 = pseudo_u64_shift(case_2, 6u32);
        case_2.1 += utils_64::extract_bits((s.head) as u64, 0u32, 6u32);
        case_2 = pseudo_u64_shift(case_2, 6u32);
        case_2.1 += utils_64::extract_bits(center_bits as u64, 0u32, 6u32);
        case_2 = pseudo_u64_shift(case_2, center_bits);
        case_2.1 += utils_64::extract_bits(xorred, (s.tail) as u32, center_bits);
        let mut case_2_bit_count = 2 + 6 + 6 + center_bits;

        // case 3: tail<=6 and lead=pr_lead
        let mut case_3: vec2<u64> = vec2::<u64>(0, 2); // code 10
        case_3 = pseudo_u64_shift(case_3, log2n);
        case_3.1 += utils_64::extract_bits(c as u64, 0u32, log2n);
        case_3 = pseudo_u64_shift(case_3, (64 - s.head) as u32);
        case_3.1 += utils_64::extract_bits(xorred, 0u32, (64 - s.head) as u32);
        let mut case_3_bit_count: u32 = 2 + 64 - (s.head) as u32;

        // case 4: tail<=6 and lead!=pr_lead
        let mut case_4: vec2<u64> = vec2::<u64>(0, 3); // code 11
        case_4 = pseudo_u64_shift(case_4, log2n);
        case_4.1 += utils_64::extract_bits(c as u64, 0u32, log2n);
        case_4 = pseudo_u64_shift(case_4, 6u32);
        case_4.1 += utils_64::extract_bits((s.head) as u64, 0u32, 6u32);
        case_4 = pseudo_u64_shift(case_4, (64 - s.head) as u32);
        case_4.1 += utils_64::extract_bits(xorred, 0u32, (64 - s.head) as u32);
        let mut case_4_bit_count: u32 = 2 + 6 + 64 - (s.head) as u32;

        let mut final_output_i32 = vec_condition(s.equal as u64) * case_1;
        final_output_i32 += vec_condition((trail_gt_6 * not_equal) as u64) * case_2;
        final_output_i32 += vec_condition((trail_le_6 * pr_lead_eq_lead) as u64) * case_3;
        final_output_i32 += vec_condition((trail_le_6 * pr_lead_ne_lead) as u64) * case_4;
        let mut final_output = vec2((final_output_i32.0), (final_output_i32.1));

        let mut final_bit_count = log2n
            + s.equal * case_1_bit_count
            + (trail_gt_6 * not_equal) * case_2_bit_count
            + (trail_le_6 * pr_lead_eq_lead) * case_3_bit_count
            + (trail_le_6 * pr_lead_ne_lead) * case_4_bit_count;
        return ChimpOutput64 {
            upper_bits: final_output.0,
            lower_bits: final_output.1,
            bit_count: (final_bit_count) as u64,
        };
    }

    pub fn new(context: Arc<Context>, n: usize) -> Self {
        Self { context, n }
    }
}
impl MaxGroupGnostic for CPUBatchedNCompressImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompressN64 for CPUBatchedNCompressImpl {
    async fn final_compress(
        &self,
        buffers: &mut RunBuffers,
        skip_time: &mut u128,
    ) -> anyhow::Result<()> {
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

        let input_staging_buffer = BufferWrapper::stage_with_size(
            self.context.device(),
            buffers.input_buffer().size() as BufferAddress,
            None,
        );
        let mut values = get_from_gpu::<f64>(
            self.context.as_ref(),
            buffers.input_buffer().buffer(),
            buffers.input_buffer().size() as BufferAddress,
            input_staging_buffer.buffer(),
        )
        .await?;
        let s_staging_buffer = BufferWrapper::stage_with_size(
            self.context.device(),
            buffers.s_buffer().size() as BufferAddress,
            None,
        );
        let mut s_values = get_from_gpu::<S>(
            self.context.as_ref(),
            buffers.s_buffer().buffer(),
            buffers.s_buffer().size() as BufferAddress,
            s_staging_buffer.buffer(),
        )
        .await?;

        let workgroup_count = self.get_max_number_of_groups(values.len());

        let mut compress = vec![ChimpOutput64::default(); values.len()];

        let chunks = ChimpBufferInfo::get().chunks();
        let iterations = workgroup_count / self.context.get_max_workgroup_size() + 1;
        let last_size = workgroup_count % self.context.get_max_workgroup_size();
        for _ in 0..iterations {
            for workgroup in (0..workgroup_count) {
                for invocation in 0..256 {
                    for i in (0..chunks) {
                        let index: usize = ((workgroup) * 256 * chunks + invocation + i * 256usize);
                        let before_index = previous_index[index + 1] as usize;
                        compress[index + 1] = self.compress(
                            s_values[index + 1],
                            values[index + 1],
                            s_values[index + 1 - before_index],
                            values[index + 1 - before_index],
                            max(1, before_index),
                        );
                    }
                }
            }
            for workgroup in (0..workgroup_count) {
                compress[(workgroup) * 256 * chunks] = ChimpOutput64 {
                    upper_bits: 0,
                    lower_bits: bytemuck::cast(values[(workgroup) * 256 * chunks]),
                    bit_count: 64u64,
                };
            }
        }

        let compress_buffer = BufferWrapper::storage_with_content(
            self.context.device(),
            bytemuck::cast_slice(&compress),
            WgpuGroupId::new(0, 0),
            Some("Input buffer"),
        );
        buffers.set_compressed_buffer(compress_buffer);

        step!(Step::Compress, {
            compress.into_iter().map(|s| format!("{}\n", s))
        });
        Ok(())
    }
}
