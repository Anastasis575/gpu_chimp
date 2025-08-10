use crate::bit_utils::ToBitVec;
use std::fmt;
use std::fmt::Formatter;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChimpOutput {
    upper_bits: u32,
    lower_bits: u32,
    bit_count: u32,
}

impl Default for ChimpOutput {
    fn default() -> Self {
        Self {
            bit_count: 2,
            lower_bits: 0,
            upper_bits: 0,
        }
    }
}

impl ChimpOutput {
    pub fn upper_bits(&self) -> u32 {
        self.upper_bits
    }

    pub fn lower_bits(&self) -> u32 {
        self.lower_bits
    }

    pub fn bit_count(&self) -> u32 {
        self.bit_count
    }

    pub fn set_upper_bits(&mut self, upper_bits: u32) {
        self.upper_bits = upper_bits;
    }

    pub fn set_lower_bits(&mut self, lower_bits: u32) {
        self.lower_bits = lower_bits;
    }

    pub fn set_bit_count(&mut self, bit_count: u32) {
        self.bit_count = bit_count;
    }
}

impl fmt::Display for ChimpOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Output:{{upper_bits:{}, lower_bits: {},bit_count: {} }}",
            self.upper_bits.to_bit_vec(),
            self.lower_bits.to_bit_vec(),
            self.bit_count
        )
    }
}


#[repr(C,packed)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ChimpOutput64 {
    upper_bits: u32,
    lower_bits: u64,
    bit_count: u32,
}

impl Default for ChimpOutput64 {
    fn default() -> Self {
        Self {
            bit_count: 2,
            lower_bits: 0,
            upper_bits: 0,
        }
    }
}

impl ChimpOutput64 {
    pub fn upper_bits(&self) -> u32 {
        self.upper_bits
    }

    pub fn lower_bits(&self) -> u64 {
        self.lower_bits
    }

    pub fn bit_count(&self) -> u32 {
        self.bit_count
    }

    pub fn set_upper_bits(&mut self, upper_bits: u32) {
        self.upper_bits = upper_bits;
    }

    pub fn set_lower_bits(&mut self, lower_bits: u64) {
        self.lower_bits = lower_bits;
    }

    pub fn set_bit_count(&mut self, bit_count: u32) {
        self.bit_count = bit_count;
    }
}

impl fmt::Display for ChimpOutput64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Output:{{upper_bits:{}, lower_bits: {},bit_count: {} }}",
            self.upper_bits().to_bit_vec(),
            self.lower_bits().to_bit_vec(),
            self.bit_count()
        )
    }
}

#[repr(C)]
#[derive(Clone, Default, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct S {
    pub head: i32,
    pub tail: i32,
    pub equal: u32,
}
impl fmt::Display for S {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "leading: {}, trailing:{}, equal: {}",
            self.head, self.tail, self.equal
        )
    }
}
