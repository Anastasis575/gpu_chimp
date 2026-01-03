
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
var<storage, read_write> input_index: array<u32>;

@group(0)
@binding(4)
var<uniform> input_size: u32;
@group(0)
@binding(5)
var<storage,read_write> last_lead_array: array<u32>;

struct CurrentInfo{
    current_index:u32,
    current_offset:u32,
}

fn write(input_idx:u32,output_idx:u32){

    //@n
    //@log2n
    
    //Index of the byte we are in
    var current_index=input_idx+1u;
    //Current Remaining offset
    var current_offset=0u;
    
    var current_info=CurrentInfo(current_index,current_offset);

    var first_num=in[current_info.current_index - 1u];
    var last_num:u32=first_num;
    var last_lead=0u;
    var significant_bits=0u;
    
    var output_index=output_idx;
    
    out[output_index]=bitcast<f32>(first_num);
    output_index+=1u;
    current_info.current_offset+=32u;

    var value=0u;
    for (var i: u32 = 1u; i < size; i++) {
        // if we have not finished reading values from the uncompressed buffers
        if current_info.current_index>=(input_size - 1u) && (current_info.current_offset - 1u) <=0u{
            break;
        }
        
        //if current bit value==1
        if get_bit_at_index(current_info.current_index,current_info.current_offset)==1u {
            current_info=decr_counter_capped_at_32(&current_info,1u);
            let recalc_lead=get_bit_at_index(current_info.current_index,current_info.current_offset)==1;
            current_info=decr_counter_capped_at_32(&current_info,1u);
            
            let compare_offset=reinterpret_num(current_info.current_index,current_info.current_offset, log2n);
            current_info=decr_counter_capped_at_32(&current_info,log2n);
            
            var last_num=bitcast<u32>(out[output_index-compare_offset]);
            var lead = last_lead_array[output_index-compare_offset];
            if  recalc_lead {
                lead = reinterpret_num(current_info.current_index,current_info.current_offset, 5u);
                current_info=decr_counter_capped_at_32(&current_info,5u);
            }
            significant_bits = 32u - lead;
            if significant_bits == 0u {
                significant_bits = 32u;
            }
            value = reinterpret_num(current_info.current_index,current_info.current_offset, u32(significant_bits));
            current_info=decr_counter_capped_at_32(&current_info,u32(significant_bits));
            value = value ^ last_num;
            last_num = value;
            last_lead_array[output_index] = lead;

            out[output_index]=bitcast<f32>(value);
            output_index+=1u;
        } else if  get_bit_at_index(current_info.current_index,current_info.current_offset - 1u)==1u{
            current_info=decr_counter_capped_at_32(&current_info,2u);
            
            let compare_offset=reinterpret_num(current_info.current_index,current_info.current_offset, log2n);
            current_info=decr_counter_capped_at_32(&current_info,log2n);
            
            var last_num=bitcast<u32>(out[output_index-compare_offset]);
            
            
            let lead = reinterpret_num(current_info.current_index,current_info.current_offset, 5u);
            current_info=decr_counter_capped_at_32(&current_info,5u);
            
            var significant_bits = reinterpret_num(current_info.current_index,current_info.current_offset, 5u);
            current_info=decr_counter_capped_at_32(&current_info,5u);
            
            if significant_bits == 0u {
                significant_bits = 32u;
            }
            
            let trail = 32u - lead - significant_bits;
            
            value = reinterpret_num(current_info.current_index,current_info.current_offset, u32(32u - lead - trail));
            
            current_info=decr_counter_capped_at_32(&current_info,u32(32u - lead - trail));
            
            value <<= trail;
            value ^= last_num;
            last_lead_array[output_index] = lead;
            last_num = value;
                            
            out[output_index]=bitcast<f32>(value);
            output_index+=1u;

        } else {
            current_info=decr_counter_capped_at_32(&current_info,2u);
            
            let compare_offset=reinterpret_num(current_info.current_index,current_info.current_offset, log2n);
            current_info=decr_counter_capped_at_32(&current_info,log2n);
            
            var last_num=bitcast<u32>(out[output_index-compare_offset]);
            var lead = last_lead_array[output_index-compare_offset];
            last_lead_array[output_index] = 32u;
            out[output_index]=bitcast<f32>(last_num);
            output_index+=1u;

        }
    }
}


fn get_bit_at_index(array_index: u32, position: u32) -> u32 {
    var index=u32(position==0u)*(array_index+1) + u32(position>0u)*array_index;
    var f_position=u32(position==0u)*32u + u32(position>0u)*position;
    return (in[index] >> (position - 1u)) & 1u;
}

fn decr_counter_capped_at_32(value:ptr<function,CurrentInfo>,count:u32)->CurrentInfo{
    let corrected_value=i32((*value).current_offset)-i32(count);
    (*value).current_offset=u32(corrected_value>0)*u32(corrected_value) + u32(corrected_value<=0)*u32(32+corrected_value);
    (*value).current_index+=u32(corrected_value<=0); //1 if it's true and 0 otherwise
    return (*value);
}

fn reinterpret_num(array_index:u32,index:u32,length:u32)->u32{
    let len=min(length,32u);
    if index>=len {
        // Fully within one u32
        return extractBits(in[array_index], u32(index-len), len);
    } else {
        // Spans two u32 elements
        let bits_in_first = index;
        let bits_in_second = length-index;

        let first_part = extractBits(in[array_index], 0u, index);
        let second_part = extractBits(in[array_index + 1], 32u - bits_in_second, bits_in_second);
        return (first_part << bits_in_second) | second_part;
    }
   
}



@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    //@workgroup_offset
    //@total_threads
    if(workgroup_offset+global_id.x>=total_threads){return;}
    write(input_index[workgroup_offset+global_id.x],(workgroup_offset+global_id.x)*size);
}