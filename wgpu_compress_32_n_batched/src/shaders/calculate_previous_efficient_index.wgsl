
struct Ss {
    leading:i32,
    trailing:i32,
    equal:u32,
}



@group(0)
@binding(1)
var<storage, read_write> input: array<f32>; // this is used as both input and output for convenience
@group(0)
@binding(2)
var<uniform> size: u32; // this is used as both input and output for convenience

//@indices_size



@group(0)
@binding(3)
var<storage, read_write> id_to_write: array<u32>; // this is used as both input and output for convenience



fn find_most_similar_previous_value(workgroup_start:u32) {
   //@n
   //@size
   //@full_size
   let log2n=log2(f32(n));
   let setlsb=u32(pow(2, log2n + 1) - 1);
   var previousIndex=1u;
   var indices=array<u32,indices_size>();
   let threshold= 5u + u32(log2n);
   for (var step=1u+workgroup_start;step<size+workgroup_start;step++){
      let value=bitcast<u32>( input[step]);
      var key=value & setlsb;
      let currIndex = indices[key];
      if (currIndex>0&&(step - currIndex) < n) {
          let tempXor = value ^ bitcast<u32>(input[currIndex]);
          let trailingZeros = countTrailingZeros(bitcast<u32>(tempXor));
      
          if (trailingZeros > threshold) {
              previousIndex = step-currIndex ;
          } else {
              previousIndex =1u;
          }
      } else {
          previousIndex =1u;
      }
      indices[key]=step;
      id_to_write[step]=previousIndex;
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
