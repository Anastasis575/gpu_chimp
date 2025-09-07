
struct Ss{
    leading:i32,
    trailing:i32,
    equal:u32,
}

@group(0)
@binding(0)
var<storage, read_write> s_store: array<Ss>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f64>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<storage, read_write> out: array<Output64>; // this is used as both input and output for convenience
@group(0)
@binding(3)
var<uniform> chunks: u32; // this is used as both input and output for convenience


//#include(64_utils)




fn compress(v:f64,s:Ss,v_prev:f64,s_prev:Ss) -> Output64{

    //Conditions
    var trail_gt_6=u32(s.trailing>6);
    var trail_le_6=u32(s.trailing<=6);
    var not_equal=u32(!bool(s.equal));
    var pr_lead=i32(s_prev.leading);
    var pr_lead_eq_lead=u32(s.leading==pr_lead);
    var pr_lead_ne_lead=u32(s.leading!=pr_lead);

    //input
    var v_prev_u64=bitcast<u64>(v_prev);
    var v_u64=bitcast<u64>(v);
    var xorred:u64= v_prev_u64^v_u64;

    var center_bits=select (64,u32(64-s.leading-s.trailing),s.leading+s.trailing<=64);


    //case 1:  xor_value=0
    var case_1:vec2<u64>=vec2<u64>(0,0);
    var case_1_bit_count:u32=2u;


    // case 2: tail>6 && xor_value!=0(!equal)
    var case_2:vec2<u64>=vec2<u64>(0,1);//code:01 bit_count=2
    case_2=pseudo_u64_shift(case_2,6u);
    case_2.y+=u64(extractBits(u32(s.leading),0u,6u));
    case_2=pseudo_u64_shift(case_2,6u);
    case_2.y+=u64(extractBits(center_bits,0u,6u));
    case_2=pseudo_u64_shift(case_2,center_bits);
    case_2.y+=extract_bits(xorred,u32(s.trailing),center_bits);
    var case_2_bit_count= 2+6+6+center_bits;

    // case 3: tail<=6 and lead=pr_lead
    var case_3:vec2<u64>=vec2<u64>(0,2); // code 10
    case_3=pseudo_u64_shift(case_3,u32(64 - s.leading));
    case_3.y+=extract_bits(xorred,0u,u32(64 - s.leading));
    var case_3_bit_count:u32=2+u32(64 - s.leading);

    // case 4: tail<=6 and lead!=pr_lead
    var case_4:vec2<u64>=vec2<u64>(0,3);// code 11
    case_4=pseudo_u64_shift(case_4,6u);
    case_4.y+=u64(extractBits(u32(s.leading),0u,6u));
    case_4=pseudo_u64_shift(case_4,u32(64 - s.leading));
    case_4.y+=extract_bits(xorred,0u,u32(64-s.leading));
    var case_4_bit_count:u32=2+6+64 - u32(s.leading);

    var final_output_i32=vec_condition(u64(s.equal))*case_1;
    final_output_i32+=vec_condition(u64(trail_gt_6*not_equal))*case_2;
    
    final_output_i32+=vec_condition(u64(trail_le_6*pr_lead_eq_lead*not_equal))*case_3;
    
    final_output_i32+= vec_condition(u64(trail_le_6*pr_lead_ne_lead*not_equal))*case_4;
 
    var final_bit_count=
        s.equal*case_1_bit_count+ 
        (trail_gt_6*not_equal)*case_2_bit_count+
        (trail_le_6*pr_lead_eq_lead*not_equal)*case_3_bit_count +
        (trail_le_6*pr_lead_ne_lead*not_equal)*case_4_bit_count;
    return Output64(final_output_i32.x,final_output_i32.y,u64(final_bit_count));
}

@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,@builtin(local_invocation_id) invocation_id: vec3<u32>) {
    //@workgroup_offset

    for (var i=0u;i<chunks;i++){
        let index:u32=(workgroup_offset+workgroup_id.x) * 256 * chunks + invocation_id.x+i*256u;
        out[index+1] = compress(in[index+1],s_store[index+1],in[index],s_store[index]);
    }
}