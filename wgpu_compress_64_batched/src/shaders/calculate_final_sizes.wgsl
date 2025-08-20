@group(0)
@binding(0)
var<storage,read_write> last_byte_index: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> in: array<Output64>; // this is used as both input and output for convenience
@group(0)
@binding(2)
var<uniform> size: u32;

//#include(64_utils)




@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    //@workgroup_offset
    var sum=u64(0);
     for (var i=0u;i<size;i++){
        sum += in[(workgroup_offset+workgroup_id.x) *size + i].bit_count;
    }
    last_byte_index[(workgroup_offset+workgroup_id.x)+1u]=u32(sum/u64(64u)) +2u;
}