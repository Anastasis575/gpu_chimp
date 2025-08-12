
struct Ss {
    leading:i32,
    trailing:i32,
    equal:u32,
}


@group(0)
@binding(0)
var<storage, read_write> s_store: array<Ss>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f64>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<uniform> chunks:u32; // how many iterations per buffer



fn calculate_s(workgoup_size:u32,id:u32,v_prev:f64,v:f64) -> Ss{
   var v_prev_u64=bitcast<u64>(v_prev);
   var v_u64=bitcast<u64>(v);
   var i:u64= v_prev_u64^v_u64;

   var leading=i32((id % workgoup_size)!=0)*i32(countLeadingZeros64(i));
   var trailing=i32(countTrailingZeros64(i));
   var equal=u32(i==0);

//   var leading_rounded:i32=i32(leading<8)*0;
//   leading_rounded+=i32(leading>=8&&leading<12)*8;
//   leading_rounded+=i32(leading>=12&&leading<16)*12;
//   leading_rounded+=i32(leading>=16&&leading<18)*16;
//   leading_rounded+=i32(leading>=18&&leading<20)*18;
//   leading_rounded+=i32(leading>=20&&leading<22)*20;
//   leading_rounded+=i32(leading>=20&&leading<24)*22;
//   leading_rounded+=i32(leading>=24)*24;

   return Ss(leading,trailing,equal);
}
fn countLeadingZeros64(x: u64) -> u32 {
    // Split into high and low 32-bit parts
    let high = u32(x>>32);
    let low = u32(x);
    
    // If high part is 0, count leading zeros in low part plus 32
    if (high == 0u) {
        return countLeadingZeros(low) + 32u;
    }
    // Otherwise, just count leading zeros in high part
    return countLeadingZeros(high);
}

fn countTrailingZeros64(x: u64) -> u32 {
    // Split into high and low 32-bit parts
    let high = u32(x>>32);
    let low = u32(x);
    
    // If low part is 0, count trailing zeros in high part plus 32
    if (low == 0u) {
        return countTrailingZeros(high) + 32u;
    }
    // Otherwise, just count trailing zeros in low part
    return countTrailingZeros(low);
}

@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,@builtin(local_invocation_id) invocation_id: vec3<u32>) {
    for (var i=0u;i<chunks;i++){
        let index:u32=workgroup_id.x * 256 * chunks + invocation_id.x+i*256u;
        s_store[index+1] = calculate_s(chunks*256,index,in[index],in[index+1]);
    }
}
