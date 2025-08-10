
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
var<storage, read_write> in: array<vec2<f32>>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<uniform> chunks:u32; // how many iterations per buffer



fn calculate_s(workgoup_size:u32,id:u32,v_prev:vec2<f32>,v:vec2<f32>) -> Ss{
   var v_prev_u64=bitcast<vec2<u32>>(v_prev);
   var v_u64=bitcast<vec2<u32>>(v);
   var i:vec2<u32>= vec2(v_prev_u64.x^v_u64.x,v_prev_u64.y^v_u64.y);

   var leading=i32((id % workgoup_size)!=0)*i32(countLeadingZeros64(i));
   var trailing=i32(countTrailingZeros64(i));
   var equal=u32(i.x==0&&i.y==0);

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
fn countLeadingZeros64(x: vec2<u32>) -> u32 {
    // Split into high and low 32-bit parts
    let high = x.x;
    let low = x.y;
    
    // If high part is 0, count leading zeros in low part plus 32
    if (high == 0u) {
        return countLeadingZeros(low) + 32u;
    }
    // Otherwise, just count leading zeros in high part
    return countLeadingZeros(high);
}

fn countTrailingZeros64(x: vec2<u32>) -> u32 {
    // Split into high and low 32-bit parts
    let high = x.x;
    let low = x.y;
    
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
    let x:u64=u64(1u)+u64(2u);
    let y:u64=u64(1u)<<2u;
    let z:u64=u64(1u)&u64(2u);
    let t:u64=bitcast<u64>(f64(1.0));
    for (var i=0u;i<chunks;i++){
        let index:u32=workgroup_id.x * 256 * chunks + invocation_id.x+i*256u;
        s_store[index+1] = calculate_s(chunks*256,index,in[index],in[index+1]);
    }
}
