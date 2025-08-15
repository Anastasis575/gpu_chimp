use crate::calculate_indexes::CalculateIndexes64;
use anyhow::Result;
use async_trait::async_trait;
use compress_utils::context::Context;
use compress_utils::general_utils::trace_steps;
use compress_utils::general_utils::Step;
use compress_utils::step;
use compress_utils::types::ChimpOutput64;
use itertools::Itertools;
use std::fs;
use std::sync::Arc;

pub struct CPUCalculateIndexes64 {
    context: Arc<Context>,
}

#[async_trait]
impl CalculateIndexes64 for CPUCalculateIndexes64 {
    async fn calculate_indexes(&self, input: &[ChimpOutput64], size: u32) -> Result<Vec<u32>> {
        let mut indexes = input
            .chunks(size as usize)
            .map(|chunk| (chunk.iter().map(|it| it.bit_count as u32).sum::<u32>() / 64u32) + 2u32)
            .collect_vec();
        (1..indexes.len()).for_each(|it| indexes[it] += indexes[it - 1]);
        indexes.insert(0, 0);
        step!(Step::CalculateIndexes, {
            indexes.iter().map(|it| format!("{it}")).into_iter()
        });
        Ok(indexes)
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
