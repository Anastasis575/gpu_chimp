
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



fn calculate_s(v_prev:f32,v:f32) -> S{
   var v_prev_u32=bitcast<u32>(v_prev);
   var v_u32=bitcast<u32>(v);
   var i= v_prev_u32^v_u32;
   var leading=i32(countLeadingZeros(i));
   var trailing=i32(countTrailingZeros(i));
   var equal=u32(i==0);
   return S(leading,trailing,equal);
}

@compute
#@workgroup_size(1)#
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    s_store[global_id.x+1] = calculate_s(in[global_id.x],in[global_id.x+1]);
}