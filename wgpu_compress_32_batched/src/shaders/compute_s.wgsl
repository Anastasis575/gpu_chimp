
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
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<uniform> chunks:u32; // how many iterations per buffer



fn calculate_s(workgoup_size:u32,id:u32,v_prev:f32,v:f32) -> Ss{
   var v_prev_u32=bitcast<u32>(v_prev);
   var v_u32=bitcast<u32>(v);
   var i= v_prev_u32^v_u32;

   var leading=i32((id % workgoup_size)!=0)*i32(countLeadingZeros(i));
   var trailing=i32(countTrailingZeros(i));
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

@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,@builtin(local_invocation_id) invocation_id: vec3<u32>) {
    //@workgroup_offset
    for (var i=0u;i<chunks;i++){
        let index:u32=(workgroup_offset+workgroup_id.x) * 256 * chunks + invocation_id.x+i*256u;
        s_store[index+1] = calculate_s(chunks*256,index,in[index],in[index+1]);
    }
}
