use crate::compute_s_shader::ComputeS;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, MaxGroupGnostic, Step};
use compress_utils::step;
use compress_utils::types::S;
use log::info;
use std::cmp::max;
use std::fs;
use std::ops::Div;
use std::sync::Arc;
use wgpu_types::BufferAddress;

pub struct CpuComputeSImpl(Arc<Context>);

impl CpuComputeSImpl {
    pub fn new(context: impl Into<Arc<Context>>) -> Self {
        Self(context.into())
    }

    pub fn context(&self) -> &Context {
        self.0.as_ref()
    }
}

impl MaxGroupGnostic for CpuComputeSImpl {
    fn get_max_number_of_groups(&self, content_len: usize) -> usize {
        content_len.div(ChimpBufferInfo::get().buffer_size())
    }
}

#[async_trait]
impl ComputeS for CpuComputeSImpl {
    async fn compute_s(&self, values: &mut [f64]) -> anyhow::Result<Vec<S>> {
        let mut Ss_vec = vec![S::default(); values.len() + 1];

        //Calculating buffer sizes and workgroup counts
        let workgroup_count = self.get_max_number_of_groups(values.len());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let size_of_s = size_of::<S>();
        let bytes = values.len() + 1;
        info!("The size of the input values vec: {}", bytes);

        let s_buffer_size = (size_of_s * bytes) as BufferAddress;
        info!("The S buffer size in bytes: {}", s_buffer_size);

        let mut padded_values = Vec::from(values);
        padded_values.push(0f64);
        let chunks = ChimpBufferInfo::get().chunks() as u32;

        for workgroup in 0u32..max(workgroup_count as u32, 1) {
            for invocation in 0u32..256u32 {
                for i in 0u32..chunks {
                    let index: usize =
                        (workgroup * 256 * chunks + invocation + i * 256u32) as usize;
                    Ss_vec[index + 1] = CpuComputeSImpl::calculate_s(
                        (chunks * 256) as u32,
                        index as u32,
                        padded_values[index],
                        padded_values[index + 1],
                    );
                }
            }
        }
        info!("Output result size: {}", Ss_vec.len());
        step!(Step::ComputeS, {
            Ss_vec
                .iter()
                .map(|it| format!("{}\n", it.to_string()))
                .into_iter()
        });
        Ok(Ss_vec)
    }
}

impl CpuComputeSImpl {
    fn calculate_s(workgoup_size: u32, id: u32, v_prev: f64, v: f64) -> S {
        let v_prev_u64: u64 = bytemuck::cast(v_prev);
        let v_u64: u64 = bytemuck::cast(v);
        let i: u64 = v_prev_u64 ^ v_u64;

        let leading =
            (((id % workgoup_size) != 0) as i32) * (CpuComputeSImpl::countLeadingZeros64(i) as i32);
        let trailing = CpuComputeSImpl::countTrailingZeros64(i) as i32;
        let equal = (i == 0) as u32;

        //   var leading_rounded:i32=i32(leading<8)*0;
        //   leading_rounded+=i32(leading>=8&&leading<12)*8;
        //   leading_rounded+=i32(leading>=12&&leading<16)*12;
        //   leading_rounded+=i32(leading>=16&&leading<18)*16;
        //   leading_rounded+=i32(leading>=18&&leading<20)*18;
        //   leading_rounded+=i32(leading>=20&&leading<22)*20;
        //   leading_rounded+=i32(leading>=20&&leading<24)*22;
        //   leading_rounded+=i32(leading>=24)*24;

        return S {
            head: leading,
            tail: trailing,
            equal,
        };
    }
    fn countLeadingZeros64(x: u64) -> u32 {
        // Split into high and low 32-bit parts
        let high: u32 = (x >> 32) as u32;
        let low = x as u32;

        // If high part is 0, count leading zeros in low part plus 32
        if high == 0u32 {
            return low.leading_zeros() + 32u32;
        }
        // Otherwise, just count leading zeros in high part
        high.leading_zeros()
    }

    fn countTrailingZeros64(x: u64) -> u32 {
        // Split into high and low 32-bit parts
        let high = (x >> 32) as u32;
        let low = x as u32;

        // If low part is 0, count trailing zeros in high part plus 32
        if low == 0u32 {
            return high.trailing_zeros() + 32u32;
        }
        // Otherwise, just count trailing zeros in low part
        low.trailing_zeros()
    }
}
