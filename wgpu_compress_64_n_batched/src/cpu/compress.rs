#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(clippy::needless_return)]
#![allow(unused_mut)]
#![allow(unused_mut)]

use crate::final_compress::FinalCompress;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::{ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::types::{ChimpOutput, S};
use compress_utils::wgpu_utils::{get_from_gpu, RunBuffers};
use compress_utils::{step, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use std::cmp::max;
use std::fs;
use std::ops::{AddAssign, Div, Mul};
use std::sync::Arc;
use wgpu_compress_32_batched::cpu::finalize::extract_bits;
use wgpu_types::BufferAddress;

pub struct vec2<T>(pub(crate) T, pub(crate) T);

pub fn vec_condition(condition: u32) -> vec2<u32> {
    vec2(condition, condition)
}

impl Mul<vec2<u32>> for vec2<u32> {
    type Output = vec2<u32>;

    fn mul(self, rhs: vec2<u32>) -> Self::Output {
        vec2(self.0 * rhs.0, self.1 * rhs.1)
    }
}
impl Mul<u32> for vec2<u32> {
    type Output = vec2<u32>;

    fn mul(self, rhs: u32) -> Self::Output {
        vec2(self.0 * (rhs as u32), self.1 * (rhs as u32))
    }
}

impl AddAssign<vec2<u32>> for vec2<u32> {
    fn add_assign(&mut self, rhs: vec2<u32>) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

pub struct CPUBatchedNCompressImpl {
    pub(crate) context: Arc<Context>,
    pub(crate) n: usize,
}

impl CPUBatchedNCompressImpl {
    pub(crate) fn compress(&self, s: S, v: f64, s_prev: S, v_prev: f64, c: usize) -> ChimpOutput {
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
        let mut v_prev_u32: u32 = bytemuck::cast(v_prev);
        let mut v_u32: u32 = bytemuck::cast(v);
        let mut xorred: u32 = v_prev_u32 ^ v_u32;

        let mut center_bits = if (s.equal != 1) {
            (32 - s.head - s.tail) as u32
        } else {
            32
        };

        //case 1:  xor_value=0
        let mut case_1: vec2<u32> = vec2(0, (c) as u32);
        let mut case_1_bit_count: u32 = 2;

        //    let mut head_representation=(s.head>=8&&s.head<12)*1+u32(s.head>=12&&s.head<16)*2+u32(s.head>=16&&s.head<18)*3+u32(s.head>=18&&s.head<20)*4+u32(s.head>=20&&s.head<22)*5+u32(s.head>=22&&s.head<24)*6+u32(s.head>=24) as u32*7;

        // case 2: tail>6 && xor_value!=0(!equal)
        let mut case_2: vec2<u32> = vec2(0u32, 1u32); //code:01 bit_count=2
        case_2 = self.pseudo_u64_shift(case_2, log2n);
        case_2.1 += extract_bits(c as u32, 0u32, log2n);
        case_2 = self.pseudo_u64_shift(case_2, 5u32);
        case_2.1 += extract_bits((s.head) as u32, 0u32, 5u32) as u32;
        case_2 = self.pseudo_u64_shift(case_2, 5u32);
        case_2.1 += extract_bits(center_bits, 0u32, 5u32);
        case_2 = self.pseudo_u64_shift(case_2, center_bits);
        case_2.1 += extract_bits(xorred, (s.tail) as u32, center_bits);
        let mut case_2_bit_count = 2 + 5 + 5 + center_bits;

        // case 3: tail<=6 and lead=pr_lead
        let mut case_3: vec2<u32> = vec2(0, 2u32); // code 10
        case_3 = self.pseudo_u64_shift(case_3, log2n);
        case_3.1 += extract_bits(c as u32, 0u32, log2n);
        case_3 = self.pseudo_u64_shift(case_3, (32 - s.head) as u32);
        case_3.1 += extract_bits(xorred, 0u32, (32 - s.head) as u32);
        let mut case_3_bit_count: u32 = 2 + 32 - (s.head) as u32;

        // case 4: tail<=6 and lead!=pr_lead
        let mut case_4: vec2<u32> = vec2(0, 3u32); // code 11
        case_4 = self.pseudo_u64_shift(case_4, log2n);
        case_4.1 += extract_bits(c as u32, 0u32, log2n);
        case_4 = self.pseudo_u64_shift(case_4, 5u32);
        case_4.1 += extract_bits((s.head) as u32, 0u32, 5u32);
        case_4 = self.pseudo_u64_shift(case_4, (32 - s.head) as u32);
        case_4.1 += extract_bits(xorred, 0u32, (32 - s.head) as u32);
        let mut case_4_bit_count: u32 = 2 + 5 + 32 - (s.head) as u32;

        let mut final_output_i32 = vec_condition(s.equal) * case_1;
        final_output_i32 += vec_condition(trail_gt_6 * not_equal) * case_2;
        final_output_i32 += vec_condition(trail_le_6 * pr_lead_eq_lead) * case_3;
        final_output_i32 += vec_condition(trail_le_6 * pr_lead_ne_lead) * case_4;
        let mut final_output = vec2((final_output_i32.0), (final_output_i32.1));

        let mut final_bit_count = log2n
            + s.equal * case_1_bit_count
            + (trail_gt_6 * not_equal) * case_2_bit_count
            + (trail_le_6 * pr_lead_eq_lead) * case_3_bit_count
            + (trail_le_6 * pr_lead_ne_lead) * case_4_bit_count;
        return ChimpOutput {
            upper_bits: final_output.0,
            lower_bits: final_output.1,
            bit_count: (final_bit_count),
        };
    }
    fn vec_condition(condition: u32) -> vec2<u32> {
        return vec2(condition, condition);
    }

    fn pseudo_u64_shift(&self, output: vec2<u32>, number: u32) -> vec2<u32> {
        let mut first_number_bits: u32 =
            wgpu_compress_32_batched::cpu::finalize::extract_bits(output.1, 32 - number, number);
        let mut new_output = vec2(output.0, output.1);
        let mut check = (number < 32);
        new_output.0 = if (check) { output.0 << number } else { 0 };
        new_output.0 += first_number_bits;
        new_output.1 = if check { output.1 << number } else { 0 };

        return new_output;
    }

    fn pseudo_u64_add(&self, output: vec2<u32>, number: u32) -> vec2<u32> {
        // check if adding <number> causes an overflow
        let mut max_u32: u32 = 0xffffffffu32;
        let mut isOverflow = (output.1 >= (max_u32 - number));
        // let mut isNotOverflow:u32=(1i32-isOverflow as i32).abs() as u32;
        let mut diff: u32 = max_u32 - output.1;

        let mut new_ouput = vec2(output.0, output.1);
        new_ouput.0 += if isOverflow { 1 } else { 0 };
        new_ouput.1 = if isOverflow {
            number - diff
        } else {
            output.1 + number
        };
        return new_ouput;
    }
}
impl MaxGroupGnostic for CPUBatchedNCompressImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompress for CPUBatchedNCompressImpl {
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

        let mut compress = vec![ChimpOutput::default(); values.len()];

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
                compress[(workgroup) * 256 * chunks] = ChimpOutput {
                    upper_bits: 0,
                    lower_bits: bytemuck::cast(values[(workgroup) * 256 * chunks]),
                    bit_count: 32u32,
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
