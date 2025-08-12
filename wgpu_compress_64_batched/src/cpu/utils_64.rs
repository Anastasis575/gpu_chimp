use std::cmp::min;
use std::ops::{AddAssign, Mul};

pub struct OutputTemp {
    x: u64,
    y: u64,
}
pub struct Output64 {
    pub(crate) upper_bits: u64,
    pub(crate) lower_bits: u64, //because there is a scenario where 32 bits are not enough to reprisent the outcome
    pub(crate) bit_count: u32,
}
#[allow(non_camel_case_types)]
pub struct vec2<T>(pub(crate) T, pub(crate) T);
pub fn pseudo_u64_shift(output: vec2<u64>, number: u32) -> vec2<u64> {
    let first_number_bits = extract_bits(output.1, 64 - number, number) as u64;
    let mut new_output = vec2(output.0, output.1);
    let mut check = ((number < 64) as u64);
    new_output.0 = if check == 1 { (output.0 << number) } else { 0 };
    new_output.0 += first_number_bits;
    new_output.1 = if check == 1 { (output.1 << number) } else { 0 };

    return new_output;
}

pub fn extract_bits(inputbits: u64, start_index: u32, bit_count: u32) -> u64 {
    let mut input_bits = inputbits;
    // assert!(start_index + bit_count > 32);
    let u32_max = 0xFFFFFFFFu32;
    let u64_max = ((u32_max as u64) << 32) + (u32_max as u64);
    let end_index: u32 = min(start_index + bit_count, 64);
    let low_bound: u64 = if start_index == 64u32 {
        0
    } else {
        u64_max << start_index
    };
    let high_bound: u64 = if (64u32 - end_index) == 64u32 {
        0
    } else {
        u64_max >> (64u32 - end_index)
    };

    input_bits = input_bits & low_bound;
    input_bits = input_bits & high_bound;
    if start_index == 64 {
        0
    } else {
        input_bits >> start_index
    }
}
pub fn insert_bits(input_bits: u64, new_bits: u64, start_index: u32, bit_count: u32) -> u64 {
    let mut output_bits = 0u64;

    let end_index = min(start_index + bit_count, 64);
    let copiable_values = end_index - start_index;

    let condition = ((copiable_values < 32) as u64);
    let bits_to_copy =
        condition * (new_bits % ((2u64) << (copiable_values - 1))) + (1 - condition) * new_bits;

    let is64 = (end_index < 32) as u64;
    output_bits += if is64 == 0 {
        0
    } else {
        input_bits >> end_index
    };
    output_bits <<= if is64 == 0 { 0 } else { copiable_values };

    output_bits += bits_to_copy;
    output_bits <<= start_index;

    let starts0 = (start_index != 0) as u64;
    output_bits += if starts0 == 0 || start_index >= 63 {
        0
    } else {
        input_bits % (2u64 << start_index)
    };
    output_bits
}

pub fn vec_condition(condition: u64) -> vec2<u64> {
    vec2(condition, condition)
}

impl Mul<vec2<u64>> for vec2<u64> {
    type Output = vec2<u64>;

    fn mul(self, rhs: vec2<u64>) -> Self::Output {
        vec2(self.0 * rhs.0, self.1 * rhs.1)
    }
}
impl Mul<u32> for vec2<u64> {
    type Output = vec2<u64>;

    fn mul(self, rhs: u32) -> Self::Output {
        vec2(self.0 * (rhs as u64), self.1 * (rhs as u64))
    }
}

impl AddAssign<vec2<u64>> for vec2<u64> {
    fn add_assign(&mut self, rhs: vec2<u64>) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}
