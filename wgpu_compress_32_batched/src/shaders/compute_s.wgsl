
struct S{
    leading:i32,
    trailing:i32,
    equal:u32,
}


@group(0)
@binding(0)
var<storage, read_write> s_store: array<S>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience



fn calculate_s(id:u32,v_prev:f32,v:f32) -> S{
   var v_prev_u32=bitcast<u32>(v_prev);
   var v_u32=bitcast<u32>(v);
   var i= v_prev_u32^v_u32;

   var leading=i32((id% @@workgroup_sizeu)!=0)*i32(countLeadingZeros(i));
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

   return S(leading,trailing,equal);
}

@compute
@workgroup_size(@@workgroup_size)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    s_store[global_id.x+1] = calculate_s(@@start_index+global_id.x,in[global_id.x],in[global_id.x+1]);
}