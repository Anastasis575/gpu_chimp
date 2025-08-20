struct Output{
    upper_bits:u32,
    lower_bits:u32,//because there is a scenario where 32 bits are not enough to reprisent the outcome
    useful_size:u32
}


@group(0)
@binding(0)
var<storage, read_write> out: array<Output>; // The compression results
@group(0)
@binding(1)
var<storage, read_write> input:array<f32>; // input values
@group(0)
@binding(2)
var<uniform> chunks:u32; // how many iterations per buffer

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    //@workgroup_offset

        out[(workgroup_offset+workgroup_id.x )* 256 * chunks]= Output(0,bitcast<u32>(input[(workgroup_offset+workgroup_id.x) * 256 * chunks]),32u);
}