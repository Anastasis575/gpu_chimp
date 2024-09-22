
struct s{
    leading:i32,
    trailing:i32
}


@group(0)
@binding(0)
var<storage, read_write> s_store: array<s>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience



fn calculate_s(v_prev:f32,v:f32) -> s{
   var i:f32= v_prev^v;
   var i_u:u32=bitcast<u32>(i);
   var leading=countLeadingZeros(i_u);
   var trailing=countTrailingZeros(i_u);
   return s(leading,trailing);
}

@compute
@workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    s_store[global_id.x] = calculate_s(v_indices[global_id.x]);
}