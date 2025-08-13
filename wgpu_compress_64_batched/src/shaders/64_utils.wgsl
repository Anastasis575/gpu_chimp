struct OutputTemp{
    x:u64,
    y:u64
}
struct Output64{
    upper_bits:u64,
    lower_bits:u64,//because there is a scenario where 32 bits are not enough to reprisent the outcome
    bit_count:u64
}

fn pseudo_u64_shift(output:vec2<u64>,number:u32)->vec2<u64>{
    var first_number_bits=u64(extract_bits(output.y,64-number,number));
    var new_output=vec2<u64>(output.x,output.y);
    var check = u64(number < 64);
    new_output.x = check*(output.x << number);
    new_output.x += first_number_bits;
    new_output.y = u64(number < 64)*(output.y<<number);
 
    return new_output;
 }
 
 fn apply_condition(input:OutputTemp,condition:u32)->OutputTemp{
     return OutputTemp(u64(condition)*input.x,u64(condition)*input.y);
 }
 
 fn add(input:OutputTemp,output:OutputTemp)->OutputTemp{
     return OutputTemp(output.x+input.x,output.y+input.y);
 }
 
 
 fn extract_bits(inputbits: u64, start_index: u32, bit_count: u32) -> u64 {
     var input_bits = inputbits;
     // assert!(start_index + bit_count > 32);
     let u32_max=0xFFFFFFFFu;
     let u64_max= (u64(u32_max)<<32) +u64(u32_max);
     let end_index:u32 = min(start_index + bit_count, 64);
     let low_bound:u64 = u64(start_index<64u)*(u64_max << start_index);
     let high_bound:u64 = u64(64u - end_index<64u)*(u64_max >> (64u - end_index));
 
     input_bits = input_bits & low_bound;
     input_bits = input_bits & high_bound;
     return u64(start_index<64u)*(input_bits >> start_index);
 }
fn insert_bits(input_bits: u64, new_bits: u64, start_index: u32, bit_count: u32) -> u64 {
    var output_bits = u64(0);

    let end_index = min(start_index + bit_count, 64);
    let copiable_values = end_index - start_index;

    let condition=u64(copiable_values < 64);
    let bits_to_copy = condition*(new_bits % (u64(2u)<<(copiable_values- 1)))+(1-condition)*new_bits;

    let is64= u64(end_index < 64);
    output_bits += u64(is64)* (input_bits >> end_index);
    output_bits <<= u32(is64)*copiable_values;
    
    output_bits += bits_to_copy;
    output_bits <<= start_index;
    
    let starts0= u64(start_index != 0);
    output_bits += u64(starts0)*u64(start_index<63)*(input_bits % u64(2u)<<start_index);
    return output_bits;
}

fn vec_condition(condition:u64)->vec2<u64>{
    return vec2<u64>(condition,condition);
}