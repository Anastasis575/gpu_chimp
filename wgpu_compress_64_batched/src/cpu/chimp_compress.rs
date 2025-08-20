use crate::cpu::utils_64;
use crate::cpu::utils_64::{pseudo_u64_shift, vec2, vec_condition};
use crate::final_compress::FinalCompress;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{ChimpBufferInfo, MaxGroupGnostic};
use compress_utils::types::{ChimpOutput64, S};
use compress_utils::wgpu_utils::RunBuffers;
use std::ops::Div;
use std::sync::Arc;
use wgpu_compress_32_batched::cpu::finalize::extract_bits;

pub struct CPUFinalCompressImpl64(Arc<Context>);

impl CPUFinalCompressImpl64 {
    pub fn new(context: Arc<Context>, _debug: bool) -> Self {
        Self(context)
    }

    pub fn context(&self) -> &Context {
        self.0.as_ref()
    }
    fn compress(v: f64, s: S, v_prev: f64, s_prev: S) -> ChimpOutput64 {
        //Conditions
        let trail_gt_6 = (s.tail > 6) as u32;
        let trail_le_6 = (s.tail <= 6) as u32;
        let not_equal = 1 - (s.equal) as u32;
        let pr_lead = s_prev.head;
        let pr_lead_eq_lead = (s.head == pr_lead) as u32;
        let pr_lead_ne_lead = (s.head != pr_lead) as u32;

        //input
        let v_prev_u64: u64 = bytemuck::cast(v_prev);
        let v_u64: u64 = bytemuck::cast(v);
        let xorred: u64 = v_prev_u64 ^ v_u64;

        let center_bits = if s.equal == 1 {
            64
        } else {
            (64 - s.head - s.tail) as u32
        };

        //case 1:  xor_value=0
        let case_1: vec2<u64> = vec2::<u64>(0, 0);
        let case_1_bit_count: u32 = 2;

        //    let mut head_representation=((s.head>=8&&s.head<12) as u32)*1+((s.head>=12&&s.head<16) as u32)*2+((s.head>=16&&s.head<18) as u32)*3+((s.head>=18&&s.head<20) as u32)*4+((s.head>=20&&s.head<22) as u32)*5+((s.head>=22&&s.head<24) as u32)*6+((s.head>=24) as u32)*7;

        // case 2: tail>6 && xor_value!=0(!equal)
        let mut case_2: vec2<u64> = vec2::<u64>(0, 1); //code:01 bit_count=2
        case_2 = pseudo_u64_shift(case_2, 6u32);
        case_2.1 += extract_bits((s.head) as u32, 0u32, 6u32) as u64;
        case_2 = pseudo_u64_shift(case_2, 6u32);
        case_2.1 += extract_bits(center_bits, 0u32, 6u32) as u64;
        case_2 = pseudo_u64_shift(case_2, center_bits);
        case_2.1 += utils_64::extract_bits(xorred, s.tail as u32, center_bits);
        let case_2_bit_count = 2 + 6 + 6 + center_bits;

        // case 3: tail<=6 and lead=pr_lead
        let mut case_3: vec2<u64> = vec2::<u64>(0, 2); // code 10
        case_3 = pseudo_u64_shift(case_3, (64 - s.head) as u32);
        case_3.1 += utils_64::extract_bits(xorred, 0u32, (64 - s.head) as u32);
        let case_3_bit_count: u32 = 2 + ((64 - s.head) as u32);

        // case 4: tail<=6 and lead!=pr_lead
        let mut case_4: vec2<u64> = vec2::<u64>(0, 3); // code 11
        case_4 = pseudo_u64_shift(case_4, 6u32);
        case_4.1 += extract_bits(s.head as u32, 0u32, 6u32) as u64;
        case_4 = pseudo_u64_shift(case_4, (64 - s.head) as u32);
        case_4.1 += utils_64::extract_bits(xorred, 0u32, (64 - s.head) as u32);
        let case_4_bit_count: u32 = 2 + 6 + 64 - ((s.head) as u32);

        let mut final_output_i32 = vec_condition(s.equal as u64) * case_1;

        final_output_i32 += vec_condition((trail_gt_6 * not_equal) as u64) * case_2;

        final_output_i32 +=
            vec_condition((trail_le_6 * pr_lead_eq_lead * not_equal) as u64) * case_3;

        final_output_i32 +=
            vec_condition((trail_le_6 * pr_lead_ne_lead * not_equal) as u64) * case_4;

        let final_bit_count = s.equal * case_1_bit_count
            + (trail_gt_6 * not_equal) * (case_2_bit_count)
            + (trail_le_6 * pr_lead_eq_lead * not_equal) * case_3_bit_count
            + (trail_le_6 * pr_lead_ne_lead * not_equal) * case_4_bit_count;
        ChimpOutput64 {
            upper_bits: final_output_i32.0,
            lower_bits: final_output_i32.1,
            bit_count: final_bit_count as u64,
        }
    }
}

impl MaxGroupGnostic for CPUFinalCompressImpl64 {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl FinalCompress for CPUFinalCompressImpl64 {
    // async fn final_compress(
    //     &self,
    //     input: &mut Vec<f64>,
    //     s_values: &mut Vec<S>,
    //     padding: usize,
    // ) -> anyhow::Result<Vec<ChimpOutput64>> {
    //     let chunks = ChimpBufferInfo::get().chunks();
    //     let mut output_vec = vec![ChimpOutput64::default(); s_values.len()];
    //     input.push(0f64);
    //     let workgroup_count = self.get_max_number_of_groups(input.len());
    //     let length_without_padding = output_vec.len() - padding - 1;
    //
    //     let mut final_output = Vec::<ChimpOutput64>::with_capacity(length_without_padding);
    //
    //     for workgroup in 0..max(workgroup_count, 1) {
    //         for invocation in 0..256 {
    //             for i in 0..chunks {
    //                 let index: usize = workgroup * 256 * chunks + invocation + i * 256usize;
    //                 output_vec[index + 1] = CPUFinalCompressImpl64::compress(
    //                     input[index + 1],
    //                     s_values[index + 1],
    //                     input[index],
    //                     s_values[index],
    //                 );
    //             }
    //         }
    //     }
    //     final_output.extend(output_vec[0..length_without_padding].to_vec());
    //     for i in 0..workgroup_count {
    //         let index = i * ChimpBufferInfo::get().buffer_size();
    //         let mut c = ChimpOutput64::default();
    //         c.set_lower_bits(bytemuck::cast(input[index]));
    //         c.set_bit_count(64);
    //
    //         final_output[index] = c;
    //     }
    //     step!(Step::Compress, {
    //         final_output
    //             .iter()
    //             .enumerate()
    //             .map(|it| format!("{}:{}\n", it.0, it.1))
    //             .into_iter()
    //     });
    //     Ok(final_output)
    // }
    async fn final_compress(&self, buffers: &mut RunBuffers) -> anyhow::Result<()> {
        // let chunks = ChimpBufferInfo::get().chunks();
        // let mut output_vec = vec![ChimpOutput64::default(); s_values.len()];
        // input.push(0f64);
        // let workgroup_count = self.get_max_number_of_groups(input.len());
        // let length_without_padding = output_vec.len() - padding - 1;
        //
        // let mut final_output = Vec::<ChimpOutput64>::with_capacity(length_without_padding);
        //
        // for workgroup in 0..max(workgroup_count, 1) {
        //     for invocation in 0..256 {
        //         for i in 0..chunks {
        //             let index: usize = workgroup * 256 * chunks + invocation + i * 256usize;
        //             output_vec[index + 1] = CPUFinalCompressImpl64::compress(
        //                 input[index + 1],
        //                 s_values[index + 1],
        //                 input[index],
        //                 s_values[index],
        //             );
        //         }
        //     }
        // }
        // final_output.extend(output_vec[0..length_without_padding].to_vec());
        // for i in 0..workgroup_count {
        //     let index = i * ChimpBufferInfo::get().buffer_size();
        //     let mut c = ChimpOutput64::default();
        //     c.set_lower_bits(bytemuck::cast(input[index]));
        //     c.set_bit_count(64);
        //
        //     final_output[index] = c;
        // }
        // step!(Step::Compress, {
        //     final_output
        //         .iter()
        //         .enumerate()
        //         .map(|it| format!("{}:{}\n", it.0, it.1))
        //         .into_iter()
        // });
        Ok(())
    }
}
