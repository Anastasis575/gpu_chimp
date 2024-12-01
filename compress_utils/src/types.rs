use crate::bit_utils::to_bit_vec;
use std::fmt;
use std::fmt::Formatter;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChimpOutput {
    content_x: u32,
    content_y: u32,
    bit_count: u32,
}

impl Default for ChimpOutput {
    fn default() -> Self {
        Self {
            bit_count: 2,
            content_y: 0,
            content_x: 0,
        }
    }
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

    pub fn set_content_x(&mut self, content_x: u32) {
        self.content_x = content_x;
    }

    pub fn set_content_y(&mut self, content_y: u32) {
        self.content_y = content_y;
    }

    pub fn set_bit_count(&mut self, bit_count: u32) {
        self.bit_count = bit_count;
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
}
