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
    useful_size:u32
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
        var overflow_bits=i32(chimp.useful_size) - 32;
        var first_add=0u;
        var second_add=0u;
        if overflow_bits>0 {
            let fitting = get_fitting(u32(overflow_bits), current_i_bits_left);
            let insert_index = get_insert_index(u32(overflow_bits), current_i_bits_left);
            let remaining = get_remaining(u32(overflow_bits), current_i_bits_left);
            bits_to_add=extractBits(chimp.upper_bits,0u,u32(overflow_bits));
            insert_index=32u-current_i_bits_left;
            out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,u32(overflow_bits));

            if current_i_bits_left<=fitting{
                current_i += 1;
                current_i_bits_left = 32u;
            }else{
                current_i_bits_left -= fitting;
            }
            if remaining>0{
                let fitting = get_fitting(remaining, current_i_bits_left);
                let insert_index = get_insert_index(remaining, current_i_bits_left);
                πg);
            }
//        }else if overflow_bits>0{
            first_add=current_i_bits_left-u32(overflow_bits);
            out[current_i]<<=first_add;
            bits_to_add=extractBits(chimp.upper_bits,u32(overflow_bits)-first_add,first_add);
            insert_index=32u-current_i_bits_left;
            out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,u32(first_add));

            current_i++;
            current_i_bits_left=32u;
            second_add=u32(overflow_bits)-first_add;
            bits_to_add=extractBits(chimp.upper_bits,0u,second_add);
            out[current_i]=insertBits(out[current_i],bits_to_add,0u,second_add);
            current_i_bits_left-=second_add;
        }
        rest_bits=min(chimp.useful_size,32u);
        rest_fit=i32(min(current_i_bits_left,rest_bits));
        out[current_i]<<=u32(rest_fit);
        bits_to_add=extractBits(chimp.lower_bits,u32(rest_bits)-u32(rest_fit),u32(rest_fit));
        insert_index=32u-current_i_bits_left;
        out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,u32(rest_fit));


        if(current_i_bits_left>rest_bits){
            current_i_bits_left-=rest_bits;
        }else{
            rest_fit=i32(rest_bits) - i32(current_i_bits_left);
            current_i_bits_left=32u;
            current_i+=1u;
            out[current_i]<<=u32(rest_fit);
            bits_to_add=extractBits(chimp.lower_bits,0u,u32(rest_fit));
            insert_index=32u-current_i_bits_left;
            out[current_i]=insertBits(out[current_i],bits_to_add,insert_index,u32(rest_fit));
        }
    }
    return current_i;

}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    last_byte_index[global_id.x]=write(global_id.x*size);
}