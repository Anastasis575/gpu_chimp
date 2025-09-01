
struct Ss {
    leading:i32,
    trailing:i32,
    equal:u32,
}


@group(0)
@binding(0)
var<storage, read_write> s_store: array<Ss>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<uniform> chunks:u32; // how many iterations per buffer

@group(0)
@binding(3)
var<storage, read_write> id_to_write: array<u32>; // this is used as both input and output for convenience



fn calculate_s(workgoup_size:u32,id:u32,v:f32) -> Ss{
   //@n
   //@full_size
   var v_u32=bitcast<u32>(v);
   var min_id=1u;
   var min_bit_count=n+1u;
   
   var new_id=id+1u;
   
   
   let start_of=new_id-(new_id%full_size);
   for (var index=1u;index<=n;index++){
       let pre_start=u32(new_id<start_of+index);
       let actual_index=u32(pre_start)*(start_of)+(1-pre_start)*(new_id-index);
       let v_prev=in[actual_index];
       var v_prev_u32=bitcast<u32>(v_prev);
       var v_xorred= v_prev_u32^v_u32;
       var leading=i32((id % workgoup_size)!=0)*i32(countLeadingZeros(v_xorred));
       var trailing=i32(countTrailingZeros(v_xorred));
       var equal=u32(v_xorred==0);
       let not_equal=u32(1-equal);
       
       let center_bits=not_equal*u32( 32u-u32(leading)-u32(trailing));
       let case_1_bit_count=2u;
       let case_2_bit_count= 2u+5u+5u+center_bits;
       let case_3_and4_bit_count:u32=2u+2u+(32u-u32(leading));//The average of the required bit

       
       let trail_gt_6=u32(trailing>6);
       let trail_le_6=u32(trailing<=6);



       var final_bit_count=equal*case_1_bit_count+
            (trail_gt_6*not_equal)*case_2_bit_count+
            (trail_le_6)*case_3_and4_bit_count;
       
       let is_min=u32(final_bit_count<min_bit_count);
       min_bit_count=u32(is_min*final_bit_count+(1-is_min)*min_bit_count);
       min_id=is_min*index+(1-is_min)*min_id;
   }
   let v_prev=in[new_id-min_id];
   var v_prev_u32=bitcast<u32>(v_prev);
   var v_xorred= v_prev_u32^v_u32;
   var leading=i32((id % workgoup_size)!=0)*i32(countLeadingZeros(v_xorred));
   var trailing=i32(countTrailingZeros(v_xorred));
   var equal=u32(v_xorred==0);
//   let not_equal=u32(v_xorred==1);
   

//   var leading_rounded:i32=i32(leading<8)*0;
//   leading_rounded+=i32(leading>=8&&leading<12)*8;
//   leading_rounded+=i32(leading>=12&&leading<16)*12;
//   leading_rounded+=i32(leading>=16&&leading<18)*16;
//   leading_rounded+=i32(leading>=18&&leading<20)*18;
//   leading_rounded+=i32(leading>=20&&leading<22)*20;
//   leading_rounded+=i32(leading>=20&&leading<24)*22;
//   leading_rounded+=i32(leading>=24)*24;
   id_to_write[new_id]=min_id;
   return Ss(leading,trailing,equal);
}

@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,@builtin(local_invocation_id) invocation_id: vec3<u32>) {
    //@workgroup_offset
    for (var i=0u;i<chunks;i++){
        let index:u32=(workgroup_offset+workgroup_id.x) * 256 * chunks + invocation_id.x+i*256u;
        s_store[index+1] = calculate_s(chunks*256,index,in[index+1]);
    }
}
