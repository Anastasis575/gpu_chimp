@group(0)
@binding(2)
var<uniform> size: u32;

@group(0)
@binding(3)
var<storage,read_write> last_byte_index: array<u32>;

@group(0)
@binding(0)
var<storage, read_write> out: array<u32>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<Output>; // this is used as both input and output for convenience
struct Output{
    upper_bits:u32,
    lower_bits:u32,//because there is a scenario where 32 bits are not enough to reprisent the outcome
    bit_count:u32
}




fn get_fitting(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
    return min(bits_rest_to_write, writeable_output_remaining);
}

fn get_remaining(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
    return max(
        bits_rest_to_write - get_fitting(bits_rest_to_write, writeable_output_remaining),
        0u,
    );
}

fn get_insert_index(bits_rest_to_write: u32, writeable_output_remaining: u32) -> u32 {
    return max(
        writeable_output_remaining - get_fitting(bits_rest_to_write, writeable_output_remaining),
        0u,
    );
}

fn write(idx:u32)->u32{
    var current_i=idx+1u;
    var current_i_bits_left=32u;
    
    var bits_to_add=0u;
    var insert_index=0u;

    var rest_bits=0u;
    var rest_fit=0;

    out[idx]=in[idx].lower_bits;
    for (var i: u32 = idx+1u; i < idx+size; i++) {
        var chimp:Output=in[i];
        var overflow_bits=i32(chimp.bit_count) - 32;
        
        var first_add=0u;
        var second_add=0u;
        
        var fitting:u32=0u;
        var insert_index:u32=0u;
        var remaining:u32=0u;
         
        var bits_to_add:u32=0u;
        
        var rest_bits:u32=0u;
         
        if overflow_bits>0 {
            fitting = get_fitting(u32(overflow_bits), current_i_bits_left);
            insert_index = get_insert_index(u32(overflow_bits), current_i_bits_left);
            remaining = get_remaining(u32(overflow_bits), current_i_bits_left);
            
            bits_to_add=extractBits(chimp.upper_bits,u32(overflow_bits-i32(fitting)),fitting);
            out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,fitting);

            if current_i_bits_left<=fitting{
                current_i += 1u;
                current_i_bits_left = 32u;
            }else{
                current_i_bits_left -= fitting;
            }
            if remaining>0{
                fitting = get_fitting(remaining, current_i_bits_left);
                insert_index = get_insert_index(remaining, current_i_bits_left);
                
                bits_to_add=extractBits(chimp.upper_bits,0u,fitting);
                out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,fitting);
                
                if current_i_bits_left<=fitting{
                    current_i += 1;
                    current_i_bits_left = 32u;
                }else{
                    current_i_bits_left -= fitting;
                }
            }
        }
        rest_bits = min(chimp.bit_count, 32u);
        fitting = get_fitting(rest_bits, current_i_bits_left);
        insert_index=get_insert_index(rest_bits, current_i_bits_left);
        remaining=get_remaining(rest_bits, current_i_bits_left);
        
        bits_to_add=extractBits(chimp.lower_bits, u32(rest_bits - fitting), fitting);
        out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,fitting);
        
        if current_i_bits_left<=fitting{
            current_i += 1u;
            current_i_bits_left = 32u;
        }else{
            current_i_bits_left -= fitting;
        }
        if remaining>0{
             fitting = get_fitting(remaining, current_i_bits_left);
             insert_index = get_insert_index(remaining, current_i_bits_left);
             bits_to_add = extractBits(chimp.lower_bits, 0u, fitting);
             out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,fitting);
             if current_i_bits_left <= fitting {
                current_i+= 1u;
                current_i_bits_left = 32u;
             } else {
                current_i_bits_left -= fitting;
             }
        }
    }
    return current_i;

}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    last_byte_index[global_id.x]=write(global_id.x*size);
}