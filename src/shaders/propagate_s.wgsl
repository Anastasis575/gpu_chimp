struct S{
    leading:i32,
    trailing:i32,
    equal:u32,
    pr_lead:u32,
}


@group(0)
@binding(0)
var<storage, read_write> s_store: array<S>; // this is used as both input and output for convenience

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_prev=max(global_id.x - 1,u32(0));
    s_store[global_id.x].pr_lead =u32(s_store[index_prev].leading);
}