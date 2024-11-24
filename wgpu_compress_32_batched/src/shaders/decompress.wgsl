
@group(0)
@binding(2)
var<uniform> size: u32;


@group(0)
@binding(3)
var<uniform> in_size: u32;

@group(0)
@binding(0)
var<storage, read_write> out: array<f32>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<u32>; // this is used as both input and output for convenience


fn write(idx:u32){
    var current_index=idx+1u;
    var current_offset=0u;


    var first_num=reinterpret_num(current_index,32);
    var last_num:u32=first_num;
    var last_lead=0u;
    
    out[0]=bitcast<f32>(first_num);
    current_index+=32u;
    for (var i: u32 = idx+1u; i < idx+size; i++) {
        if current_index==in_size - 1 && current_offset + 1 >= 32{
            break;
        }
        if get_bit_at_index(current_index,current_offset) {
                        input_index += 1;
                        let mut lead = last_lead;
                        if input_vector[input_index] {
                            input_index += 1;
                            lead = input_vector.reinterpret_u32(input_index, 5);
                            input_index += 5;
                        } else {
                            input_index += 1;
                        }
                        let mut significant_bits = 32 - lead;
                        if significant_bits == 0 {
                            significant_bits = 32;
                        }
                        let value = input_vector.reinterpret_u32(input_index, significant_bits as usize);
                        input_index += (32 - lead) as usize;
                        let value = value ^ last_num;
                        last_num = value;
                        last_lead = lead;

                        if value == u32::MAX {
                            break;
                        } else {
                            let value_f32 = f32::from_bits(value);
                            if self.debug {
                                log::info!("{}:{}", output.len(), value_f32);
                            }
                            output.push(value_f32);
                        }
                    } else if input_vector[input_index + 1] {
                        input_index += 2;
                        let lead = input_vector.reinterpret_u32(input_index, 5);
                        input_index += 5;
                        let mut significant_bits = input_vector.reinterpret_u32(input_index, 5);
                        input_index += 5;
                        if significant_bits == 0 {
                            significant_bits = 32;
                        }
                        let trail = 32 - lead - significant_bits;
                        let mut value =
                            input_vector.reinterpret_u32(input_index, (32 - lead - trail) as usize);
                        input_index += (32 - lead - trail) as usize;
                        value <<= trail;
                        value ^= last_num;
                        last_lead = lead;
                        last_num = value;
                        if value == u32::MAX {
                            break;
                        } else {
                            let value_f32 = f32::from_bits(value);
                            if self.debug {
                                log::info!("{}:{}", output.len(), value_f32);
                            }
                            output.push(value_f32);
                        }
                    } else {
                        let value_f32 = f32::from_bits(last_num);
                        last_lead = 32;
                        if self.debug {
                            log::info!("{}:{}", output.len(), value_f32);
                        }
                        output.push(value_f32);
                        input_index += 2;
                    }
                }
    }
}
fn get_bit_at_index(current_i:u32,offset:u32)->u32{
    return extractBits(in[current_i],32u-offset - 1u,1u)
}
fn reinterpret_num(index:u32,length:u32){

}



@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    write(global_id.x*size);
}