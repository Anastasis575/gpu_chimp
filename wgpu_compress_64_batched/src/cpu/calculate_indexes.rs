use crate::calculate_indexes::CalculateIndexes64;
use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::wgpu_utils::RunBuffers;
use std::sync::Arc;

pub struct CPUCalculateIndexes64 {
    context: Arc<Context>,
}

#[async_trait]
impl CalculateIndexes64 for CPUCalculateIndexes64 {
    async fn calculate_indexes(&self, input: &mut RunBuffers, size: u32) -> Result<()> {
        // let mut indexes = input
        //     .chunks(size as usize)
        //     .map(|chunk| (chunk.iter().map(|it| it.bit_count as u32).sum::<u32>() / 64u32) + 2u32)
        //     .collect_vec();
        // (1..indexes.len()).for_each(|it| indexes[it] += indexes[it - 1]);
        // indexes.insert(0, 0);
        // step!(Step::CalculateIndexes, {
        //     indexes.iter().map(|it| format!("{it}")).into_iter()
        // });
        Ok(())
    }
}
impl CPUCalculateIndexes64 {
    pub fn new(context: Arc<Context>) -> Self {
        Self { context }
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }
}
