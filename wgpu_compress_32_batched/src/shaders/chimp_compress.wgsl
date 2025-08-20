
struct Ss{
    leading:i32,
    trailing:i32,
    equal:u32,
}

struct Output{
    upper_bits:u32,
    lower_bits:u32,//because there is a scenario where 32 bits are not enough to reprisent the outcome
    useful_size:u32
}

@group(0)
@binding(0)
var<storage, read_write> s_store: array<Ss>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<storage, read_write> out: array<Output>; // this is used as both input and output for convenience
@group(0)
@binding(3)
var<uniform> chunks: u32; // this is used as both input and output for convenience



fn compress(v:f32,s:Ss,v_prev:f32,s_prev:Ss) -> Output{

    //Conditions
    var trail_gt_6=u32(s.trailing>6);
    var trail_le_6=u32(s.trailing<=6);
    var not_equal=u32(!bool(s.equal));
    var pr_lead=u32(s_prev.leading);
    var pr_lead_eq_lead=u32(s.leading==i32(pr_lead));
    var pr_lead_ne_lead=u32(s.leading!=i32(pr_lead));

    //Constants
    //0x1000 0000 0000 0000 0000 0000 0000 0000
    var first_bit_one:u32=0x80000000;

    //input
    var v_prev_u32=bitcast<u32>(v_prev);
    var v_u32=bitcast<u32>(v);
    var xorred:u32= v_prev_u32^v_u32;

    var center_bits=u32(32-s.leading-s.trailing);

    //Output
    var content:vec2<u32>=vec2(0,0);
    var bit_count:u32=0;

    //case 1:  xor_value=0
    var case_1:vec2<u32>=vec2(0,0);
    var case_1_bit_count:u32=2;

//    var leading_representation=u32(s.leading>=8&&s.leading<12)*1+u32(s.leading>=12&&s.leading<16)*2+u32(s.leading>=16&&s.leading<18)*3+u32(s.leading>=18&&s.leading<20)*4+u32(s.leading>=20&&s.leading<22)*5+u32(s.leading>=22&&s.leading<24)*6+u32(s.leading>=24)*7;


    // case 2: tail>6 && xor_value!=0(!equal)
    var case_2:vec2<u32>=vec2(0u,1u);//code:01 bit_count=2
    case_2=pseudo_u64_shift(case_2,5u);
    case_2.y+=extractBits(u32(s.leading),0u,5u);
    case_2=pseudo_u64_shift(case_2,5u);
    case_2.y+=extractBits(center_bits,0u,5u);
    case_2=pseudo_u64_shift(case_2,center_bits);
    case_2.y+=extractBits(xorred,u32(s.trailing),center_bits);
    var case_2_bit_count= 2+5+5+center_bits;

    // case 3: tail<=6 and lead=pr_lead
    var case_3:vec2<u32>=vec2(0,2); // code 10
    case_3=pseudo_u64_shift(case_3,u32(32 - s.leading));
    case_3.y+=extractBits(xorred,0u,u32(32 - s.leading));
    var case_3_bit_count:u32=2+32-u32(s.leading);

    // case 4: tail<=6 and lead!=pr_lead
    var case_4:vec2<u32>=vec2(0,3);// code 11
    case_4=pseudo_u64_shift(case_4,5u);
    case_4.y+=extractBits(u32(s.leading),0u,5u);
    case_4=pseudo_u64_shift(case_4,u32(32 - s.leading));
    case_4.y+=extractBits(xorred,0u,u32(32-s.leading));
    var case_4_bit_count:u32=2+5+32 - u32(s.leading);

    var final_output_i32=vec_condition(s.equal)*case_1;
    final_output_i32+= vec_condition(trail_gt_6*not_equal)*case_2;
    final_output_i32+= vec_condition(trail_le_6*pr_lead_eq_lead)*case_3;
    final_output_i32+=vec_condition(trail_le_6*pr_lead_ne_lead)*case_4;
    var final_output=vec2(u32(final_output_i32.x),u32(final_output_i32.y));

    var final_bit_count=s.equal*case_1_bit_count+ (trail_gt_6*not_equal)*case_2_bit_count +(trail_le_6*pr_lead_eq_lead)*case_3_bit_count +(trail_le_6*pr_lead_ne_lead)*case_4_bit_count;
    return Output(final_output.x,final_output.y,u32(final_bit_count));
}

fn vec_condition(condition:u32)->vec2<u32>{
    return vec2(condition,condition);
}

fn pseudo_u64_shift(output:vec2<u32>,number:u32)->vec2<u32>{
    var first_number_bits:u32=extractBits(output.y,32-number,number);
    var new_output=vec2(output.x,output.y);
    var check = u32(number < 32);
    new_output.x = check*(output.x << number);
    new_output.x += first_number_bits;
    new_output.y = check*(output.y<<number);

    return new_output;
}


fn pseudo_u64_add(output:vec2<u32>,number:u32)->vec2<u32>{
    // check if adding <number> causes an overflow
    var max_u32:u32=0xffffffffu;
    var isOverflow:u32=u32(output.y>=(max_u32-number));
    var isNotOverflow:u32=u32(abs(1-isOverflow));
    var diff:u32=max_u32-output.y;

    var new_ouput=vec2(output.x,output.y);
    new_ouput.x+=1*isOverflow;
    new_ouput.y=isOverflow*(number-diff) + isNotOverflow*(output.y+number);
    return new_ouput;
}

@compute
@workgroup_size(256)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,@builtin(local_invocation_id) invocation_id: vec3<u32>) {
    for (var i=0u;i<chunks;i++){
        let index:u32=workgroup_id.x * 256 * chunks + invocation_id.x+i*256u;
        out[index+1] = compress(in[index+1],s_store[index+1],in[index],s_store[index]);
    }
}