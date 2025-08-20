
@group(0)
@binding(0)
var<storage, read_write> out: array<Output64>; // The compression results
@group(0)
@binding(1)
var<storage, read_write> input:array<f64>; // input values
@group(0)
@binding(2)
var<uniform> chunks:u32; // how many iterations per buffer
//#include(64_utils)
@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    out[workgroup_id.x * 256 * chunks]= Output64(u64(0u),bitcast<u64>(input[workgroup_id.x * 256 * chunks]),u64(64u));
}