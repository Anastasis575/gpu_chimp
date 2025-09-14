
struct Ss {
    leading:i32,
    trailing:i32,
    equal:u32,
}



@group(0)
@binding(1)
var<storage, read_write> input: array<f64>; // this is used as both input and output for convenience
@group(0)
@binding(2)
var<uniform> size: u32; // this is used as both input and output for convenience

//@indices_size

//#include(64_utils)
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
   let threshold= 6 + u32(log2n);
   for (var step=1u+workgroup_start;step<size+workgroup_start;step++){
      let value=bitcast<u64>(input[step]);
      var key=u32(value) & setlsb;
      let currIndex = indices[key];
      if (currIndex>0&&(step - currIndex) < n) {
          let tempXor = value ^ u64(input[currIndex]);
          let trailingZeros = countLeadingZeros64(tempXor);
      
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
fn countLeadingZeros64(x: u64) -> u32 {
    // Split into high and low 32-bit parts
    let high = u32(x>>32);
    let low = u32(x);
    
    // If high part is 0, count leading zeros in low part plus 32
    if (high == 0u) {
        return countLeadingZeros(low) + 32u;
    }
    // Otherwise, just count leading zeros in high part
    return countLeadingZeros(high);
}

fn countTrailingZeros64(x: u64) -> u32 {
    // Split into high and low 32-bit parts
    let high = u32(x>>32);
    let low = u32(x);
    
    // If low part is 0, count trailing zeros in high part plus 32
    if (low == 0u) {
        return countTrailingZeros(high) + 32u;
    }
    // Otherwise, just count trailing zeros in low part
    return countTrailingZeros(low);
}

@compute
@workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    //@workgroup_offset
    find_most_similar_previous_value((workgroup_offset+workgroup_id.x)*size);
}
