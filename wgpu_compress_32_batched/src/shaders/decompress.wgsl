
@group(0)
@binding(0)
var<storage, read_write> out: array<f32>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<u32>; // this is used as both input and output for convenience


@group(0)
@binding(2)
var<uniform> size: u32;


@group(0)
@binding(3)
var<uniform> in_size: u32;

fn write(idx:u32){
    var current_index=idx+1u;
    var current_offset=0u;

    var first_num=reinterpret_num(current_index,32);
    var last_num:u32=first_num;
    var last_lead=0u;
    
    var output_index=0u;
    
    out[output_index]=bitcast<f32>(first_num);
    current_index+=32u;
    for (var i: u32 = idx+1u; i < idx+size; i++) {
        if current_index==in_size - 1u && current_offset + 1u >= 32u{
            break;
        }
        if get_bit_at_index(current_index,current_offset)==1 {
            input_index += 1u;
            var lead = last_lead;
            if  get_bit_at_index(current_index,current_offset)==1 {
                input_index += 1u;
                lead = reinterpret_u32(input_index, 5u);
                input_index += 5u;
            } else {
                input_index += 1u;
            }
            let significant_bits = 32u - lead;
            if significant_bits == 0u {
                significant_bits = 32u;
            }
            let value = reinterpret_u32(input_index, u32(significant_bits));
            input_index += u32(32u - lead);
            let value = value ^ last_num;
            last_num = value;
            last_lead = lead;

            out[output_index]=bitcast<f32>(value);
        } else if  get_bit_at_index(input_index + 1u,current_offset)==1u{
            input_index += 2u;
            let lead = reinterpret_u32(input_vectorinput_index, 5u);
            input_index += 5u;
            var significant_bits = reinterpret_u32(input_vectorinput_index, 5u);
            input_index += 5u;
            if significant_bits == 0u {
                significant_bits = 32u;
            }
            let trail = 32u - lead - significant_bits;
            var value = reinterpret_u32(input_vectorinput_index, u32(32u - lead - trail));
            input_index += u32(32u - lead - trail);
            value <<= trail;
            value ^= last_num;
            last_lead = lead;
            last_num = value;
                            
            out[output_index]=bitcast<f32>(value);
            output_index+=1u;

        } else {
            out[output_index]=bitcast<f32>(value);
            output_index+=1u;

            last_lead = 32u;
            input_index += 2u;
        }
    }
}

fn get_bit_at_index(current_i:u32,offset:u32)->u32{
    return extractBits(in[current_i],32u-offset - 1u,1u);
}
fn reinterpret_num(index:u32,length:u32)->u32{
    return extractBits(in,index,length);
}



@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    write(global_id.x*size);
}