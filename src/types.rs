use crate::utils::bit_utils::to_bit_vec;
use std::fmt;
use std::fmt::Formatter;

#[repr(C)]
#[derive(Clone, Default, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChimpOutput {
    content_x: u32,
    content_y: u32,
    bit_count: u32,
}

impl ChimpOutput {
    pub fn content_x(&self) -> u32 {
        self.content_x
    }

    pub fn content_y(&self) -> u32 {
        self.content_y
    }

    pub fn bit_count(&self) -> u32 {
        self.bit_count
    }
}

impl fmt::Display for ChimpOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Output:{{content_x:{}, content_y: {},bit_count: {} }}",
            to_bit_vec(self.content_x),
            to_bit_vec(self.content_y),
            self.bit_count
        )
    }
}

#[repr(C)]
#[derive(Clone, Default, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct S {
    head: i32,
    tail: i32,
    equal: u32,
    pr_lead: u32,
}
