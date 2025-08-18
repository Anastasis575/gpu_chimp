use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::{trace_steps, ChimpBufferInfo, CompressResult, Step};
use compress_utils::types::ChimpOutput64;
use compress_utils::{execute_compute_shader, step, wgpu_utils, BufferWrapper, WgpuGroupId};
use itertools::Itertools;
use log::info;
use std::cmp::{max, min};
use std::ops::Div;
use std::sync::Arc;
use std::{fs, vec};
use wgpu_types::BufferAddress;

#[async_trait]
pub trait Finalize {
    async fn finalize(
        &self,
        chimp_output: &mut Vec<ChimpOutput64>,
        padding: usize,
        indexes: Vec<u32>,
    ) -> Result<CompressResult>;
}

#[derive(Debug)]
pub struct Finalizer64 {
    context: Arc<Context>,
}

impl Finalizer64 {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }
    pub fn context(&self) -> &Context {
        &self.context
    }
}

#[async_trait]
impl Finalize for Finalizer64 {
    async fn finalize(
        &self,
        chimp_input: &mut Vec<ChimpOutput64>,
        padding: usize,
        indexes: Vec<u32>,
    ) -> Result<CompressResult> {
        let util_64 = include_str!("shaders/64_utils.wgsl");
        let temp = include_str!("shaders/chimp_finalize_compress.wgsl")
            .replace("//#include(64_utils)", util_64)
            .to_string();

        // let size_of_chimp = size_of::<ChimpOutput>();
        let size_of_out = size_of::<u64>();

        let chimp_input_length = chimp_input.len() - padding;
        let size = ChimpBufferInfo::get().buffer_size() as u32;
        let last_size = if (chimp_input_length % ChimpBufferInfo::get().buffer_size() == 0) {
            ChimpBufferInfo::get().buffer_size()
        } else {
            (chimp_input_length % ChimpBufferInfo::get().buffer_size()) - 1
        } as u32;

        let input_length = chimp_input_length;
        info!("The length of the input vec: {}", input_length);

        let output_buffer_size =
            ((*indexes.last().unwrap() + 1) as usize * size_of::<u64>()) as BufferAddress;
        info!("The Output buffer size in bytes: {}", &output_buffer_size);

        let workgroup_count = chimp_input.len().div(ChimpBufferInfo::get().buffer_size());
        info!("The wgpu workgroup size: {}", &workgroup_count);

        let out_stage_buffer = BufferWrapper::stage_with_size(
            self.context().device(),
            output_buffer_size,
            Some("Staging Output Buffer"),
        );
        let out_storage_buffer = BufferWrapper::storage_with_size(
            self.context().device(),
            output_buffer_size,
            WgpuGroupId::new(0, 0),
            Some("Staging Output Buffer"),
        );
        let in_storage_buffer = BufferWrapper::storage_with_content(
            self.context().device(),
            bytemuck::cast_slice(chimp_input.as_slice()),
            WgpuGroupId::new(0, 1),
            Some("Staging Output Buffer"),
        );
        let size_uniform = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&size),
            WgpuGroupId::new(0, 2),
            Some("Size Uniform Buffer"),
        );
        let last_size_uniform = BufferWrapper::uniform_with_content(
            self.context().device(),
            bytemuck::bytes_of(&last_size),
            WgpuGroupId::new(0, 4),
            Some("Last buffer size Uniform Buffer"),
        );

        let useful_byte_count_storage = BufferWrapper::storage_with_content(
            self.context().device(),
            bytemuck::cast_slice(&indexes),
            WgpuGroupId::new(0, 3),
            Some("Useful Storage Buffer"),
        );

        execute_compute_shader!(
            self.context(),
            &temp,
            vec![
                &out_stage_buffer,
                &out_storage_buffer,
                &in_storage_buffer,
                &size_uniform,
                &useful_byte_count_storage,
                &last_size_uniform,
            ],
            workgroup_count,
            Some("trim pass")
        );

        let output = wgpu_utils::get_s_output::<u8>(
            self.context(),
            out_storage_buffer.buffer(),
            output_buffer_size,
            out_stage_buffer.buffer(),
        )
        .await?;

        // let indexes = wgpu_utils::get_s_output::<u32>(
        //     self.context(),
        //     useful_byte_count_storage.buffer(),
        //     (workgroup_count * size_of::<u32>()) as BufferAddress,
        //     useful_byte_count_staging.buffer(),
        // )
        // .await?;

        let mut final_vec = output;

        // for (i, useful_byte_count) in indexes.iter().enumerate() {
        //     let start_index = i * ChimpBufferInfo::get().buffer_size();
        //     let byte_count = min(*useful_byte_count as usize, chimp_input_length - 1);
        //     let temp_vec = output[start_index..=byte_count]
        //         .iter()
        //         .flat_map(|it| it.to_be_bytes())
        //         .collect_vec();
        //
        //     let batch_size = if i == workgroup_count - 1
        //         && chimp_input_length % ChimpBufferInfo::get().buffer_size() != 0
        //     {
        //         ((chimp_input_length % ChimpBufferInfo::get().buffer_size()) - 1) as u32
        //     } else {
        //         (ChimpBufferInfo::get().buffer_size() - 1) as u32
        //     };
        //     final_vec.extend(batch_size.to_be_bytes());
        //     final_vec.extend((temp_vec.len() as u32).to_be_bytes().iter());
        //     final_vec.extend(temp_vec);
        // }
        step!(&Step::Finalize, {
            final_vec
                .iter()
                .chunks(8)
                .into_iter()
                .map(|chunk| chunk.map(|it| format!("{:08b}", it)).join(" ") + "\n")
                .collect_vec()
                .into_iter()
        });
        Ok(CompressResult(final_vec, workgroup_count * 8))
    }
}
