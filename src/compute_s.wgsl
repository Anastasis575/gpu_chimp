
struct S{
    leading:i32,
    trailing:i32,
    equal:bool
}
struct Output{
    compressed:f32,
    size:u32
}


@group(0)
@binding(0)
var<storage, read_write> s_store: array<S>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience



fn calculate_s(v_prev:f32,v:f32) -> S{
   var i:f32= v_prev^v;
   var i_u:u32=bitcast<u32>(i);
   var leading=countLeadingZeros(i_u);
   var trailing=countTrailingZeros(i_u);
   var equal=i=0
   return s(leading,trailing,equal);
}

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_prev=max(global_id.x,0)
    s_store[global_id.x] = calculate_s(in[global_id.x],in[index_prev]);
}