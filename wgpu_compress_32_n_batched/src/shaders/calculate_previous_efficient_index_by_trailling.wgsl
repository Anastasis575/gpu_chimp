
struct Ss {
    leading:i32,
    trailing:i32,
    equal:u32,
}



@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience
@group(0)
@binding(2)
var<uniform> size: u32; // this is used as both input and output for convenience





@group(0)
@binding(3)
var<storage, read_write> id_to_write: array<u32>; // this is used as both input and output for convenience



fn find_most_similar_previous_value(workgroup_start:u32) {
   //@n
   //@size
   //@full_size
   for (var step=1u+workgroup_start;step<=size+workgroup_start;step++){
       var v_u32=bitcast<u32>(in[step]);
       var min_id=1u;
       var max_trail=0u;
       
       var new_id=step;
       
       
       for (var index=1u;index<=n;index++){
           let pre_start=u32(new_id<workgroup_start+index);
           let actual_index=u32(pre_start)*(workgroup_start)+(1-pre_start)*(new_id-index);
           let v_prev=in[actual_index];
           var v_prev_u32=bitcast<u32>(v_prev);
           var v_xorred= v_prev_u32^v_u32;
           var trailing=u32(countTrailingZeros(v_xorred));
           min_id=u32(trailing>max_trail)*index+u32(trailing<=max_trail)*min_id;
           max_trail=max(trailing,max_trail);
       }
       id_to_write[new_id]=min_id;
   }
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    //@workgroup_offset
//    for (var i=0u;i<chunks;i++){
//        let index:u32=(workgroup_offset+workgroup_id.x) * 256 * chunks + invocation_id.x+i*256u;
        find_most_similar_previous_value((workgroup_offset+workgroup_id.x)*size);
//    }
}
