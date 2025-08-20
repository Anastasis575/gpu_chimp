@group(0)
@binding(0)
var<storage,read_write> last_byte_index: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> in: array<Output>; // this is used as both input and output for convenience
@group(0)
@binding(2)
var<uniform> size: u32;
struct Output{
    upper_bits:u32,
    lower_bits:u32,//because there is a scenario where 32 bits are not enough to reprisent the outcome
    useful_size:u32
}





@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    var sum=0u;
     for (var i=0u;i<size;i++){
        sum += in[workgroup_id.x *size + i].useful_size;
    }
    last_byte_index[workgroup_id.x+1]=u32(sum/32) +3u;
}