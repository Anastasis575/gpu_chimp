
struct Ss{
    leading:i32,
    trailing:i32,
    equal:u32,
}

struct Output{
    upper_bits:u32,
    lower_bits:u64, //because there is a scenario where 32 bits are not enough to reprisent the outcome
    useful_size:u32
}

@group(0)
@binding(0)
var<storage, read_write> s_store: array<Ss>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f64>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<storage, read_write> out: array<Output>; // this is used as both input and output for convenience
@group(0)
@binding(3)
var<uniform> chunks: u32; // this is used as both input and output for convenience

struct OutputTemp{
    x:u32,
    y:u64
}

fn compress(v:f64,s:Ss,v_prev:f64,s_prev:Ss) -> Output{

    //Conditions
    var trail_gt_6=u32(s.trailing>6);
    var trail_le_6=u32(s.trailing<=6);
    var not_equal=u32(!bool(s.equal));
    var pr_lead=u32(s_prev.leading);
    var pr_lead_eq_lead=u32(s.leading==i32(pr_lead));
    var pr_lead_ne_lead=u32(s.leading!=i32(pr_lead));

    //input
    var v_prev_u64=bitcast<u64>(v_prev);
    var v_u64=bitcast<u64>(v);
    var xorred:u64= v_prev_u64^v_u64;

    var center_bits=u32(64-s.leading-s.trailing);

    //Output
    var content:OutputTemp=OutputTemp(0,0);
    var bit_count:u32=0;

    //case 1:  xor_value=0
    var case_1:OutputTemp=OutputTemp(0,0);
    var case_1_bit_count:u32=2;

//    var leading_representation=u32(s.leading>=8&&s.leading<12)*1+u32(s.leading>=12&&s.leading<16)*2+u32(s.leading>=16&&s.leading<18)*3+u32(s.leading>=18&&s.leading<20)*4+u32(s.leading>=20&&s.leading<22)*5+u32(s.leading>=22&&s.leading<24)*6+u32(s.leading>=24)*7;


    // case 2: tail>6 && xor_value!=0(!equal)
    var case_2:OutputTemp=OutputTemp(0,1);//code:01 bit_count=2
    case_2=pseudo_u64_shift(case_2,5u);
    case_2.y+=u64(extractBits(u32(s.leading),0u,5u));
    case_2=pseudo_u64_shift(case_2,5u);
    case_2.y+=u64(extractBits(center_bits,0u,5u));
    case_2=pseudo_u64_shift(case_2,center_bits);
    case_2.y+=extractBits(xorred,u32(s.trailing),center_bits);
    var case_2_bit_count= 2+5+5+center_bits;

    // case 3: tail<=6 and lead=pr_lead
    var case_3:OutputTemp=OutputTemp(0,2); // code 10
    case_3=pseudo_u64_shift(case_3,u32(64 - s.leading));
    case_3.y+=extractBits(xorred,0u,u32(64 - s.leading));
    var case_3_bit_count:u32=2+64-u32(s.leading);

    // case 4: tail<=6 and lead!=pr_lead
    var case_4:OutputTemp=OutputTemp(0,3);// code 11
    case_4=pseudo_u64_shift(case_4,5u);
    case_4.y+=u64(extractBits(u32(s.leading),0u,5u));
    case_4=pseudo_u64_shift(case_4,u32(64 - s.leading));
    case_4.y+=extractBits(xorred,0u,u32(64-s.leading));
    var case_4_bit_count:u32=2+5+64 - u32(s.leading);

    var final_output_i32=vec_condition(s.equal)*case_1;
    final_output_i32+= vec_condition(trail_gt_6*not_equal)*case_2;
    final_output_i32+= vec_condition(trail_le_6*pr_lead_eq_lead)*case_3;
    final_output_i32+=vec_condition(trail_le_6*pr_lead_ne_lead)*case_4;
    var final_output=OutputTemp(u32(final_output_i32.x),u32(final_output_i32.y));

    var final_bit_count=s.equal*case_1_bit_count+ (trail_gt_6*not_equal)*case_2_bit_count +(trail_le_6*pr_lead_eq_lead)*case_3_bit_count +(trail_le_6*pr_lead_ne_lead)*case_4_bit_count;
    return Output(final_output.upper_bits,final_output.low_bits,u32(final_bit_count));
}

fn vec_condition(condition:u32)->OutputTemp{
    return OutputTemp(condition,condition);
}

fn pseudo_u64_shift(output:OutputTemp,number:u32)->OutputTemp{
   var first_number_bits:u32=u32(extractBits(output.y,64-number,number));
   var new_output=OutputTemp(output.x,output.y);
   var check = u32(number < 64);
   new_output.x = check*(output.x << number);
   new_output.x += first_number_bits;
   new_output.y = check*(output.y<<number);

   return new_output;
}


fn extract_bits(inputbits: u64, start_index: u32, bit_count: u32) -> u32 {
    var input_bits = input_bits;
    // assert!(start_index + bit_count > 32);
    let u32_max=0xFFFFFFFF;
    let u64_max= u64(u32_max)<<32 +u32_max;
    let end_index:u64 = min(start_index + bit_count, 64);
    let low_bound:u64 = u64_max << start_index;
    let high_bound:u64 = u64_max >> (64 - end_index);

    input_bits = input_bits & low_bound;
    input_bits = input_bits & high_bound;
    return input_bits >> start_index;
}
@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,@builtin(local_invocation_id) invocation_id: vec3<u32>) {
     for (var i=0u;i<chunks;i++){
        let index:u32=workgroup_id.x * 256 * chunks + invocation_id.x+i*256u;
        out[index+1] = compress(in[index+1],s_store[index+1],in[index],s_store[index]);
    }
}