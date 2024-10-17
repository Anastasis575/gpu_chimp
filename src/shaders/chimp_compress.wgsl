
struct S{
    leading:i32,
    trailing:i32,
    equal:u32,
    pr_lead:u32,
}

struct Output{
    content:vec2<u32>,//because there is a scenario where 32 bits are not enough to reprisent the outcome
    useful_size:u32
}

@group(0)
@binding(0)
var<storage, read_write> s_store: array<S>; // this is used as both input and output for convenience

@group(0)
@binding(1)
var<storage, read_write> in: array<f32>; // this is used as both input and output for convenience

@group(0)
@binding(2)
var<storage, read_write> out: array<Output>; // this is used as both input and output for convenience




fn compress(v_prev:f32,s_prev:S,v:f32,s:S) -> Output{
    //Conditions
    var trail_gt_6=u32(s.trailing>6);
    var trail_le_6=u32(s.trailing<=6);
    var not_equall=u32(!s.equal);
    var pr_lead_eq_lead=u32(s.leading==s.pr_lead);
    var pr_lead_ne_lead=u32(s.leading!=s.pr_lead);

    //Constants
    //0x1000 0000 0000 0000 0000 0000 0000 0000
    var first_bit_one:vec2<u32>=0x80000000;

    //input
    var v_prev_u32=bitcast<u32>(v_prev);
    var v_u32=bitcast<u32>(v);
    var xorred= v_prev_u32^v_u32;

    var center_bits=32-s.leading-s.trailing;

    //Output
    var content:vec2<u32>=vec2(0,0);
    var bit_count:u32=0;

    //case 1:  xor_value=0
    var case_1=vec2(0,0);
    var case_1_bit_count=2;

    // case 2: tail>6 && xor_value!=0(!equal)
    var case_2:vec2<u32>=vec2(0,1);//code:01 bit_count=2
    case_2=pseudo_u64_shift(case_2,3);
    case_2=pseudo_u64_add(case_2,s.leading%3);
    case_2=pseudo_u64_shift(case_2,6);
    case_2=pseudo_u64_add(cas3_2,center_bits%6);
    case_2=pseudo_u64_shift(case2,center_bits);
    case_2= pseudo_u64_add(case_2,(xorred>>s.trailing)%center_bits);
    var case_2_bit_count= 2+3+6+center_bits;

    // case 3: tail<=6 and lead=pr_lead
    var case_3:vec2<u32>=vec2(0,2); // code 10
    case_3=pseudo_u64_shift(case_3,32 - s.leading);
    case_3=pseudo_u64_add(case_3,xorred%(32-s.leading));
    var case_3_bit_count:u32=2+32 - s.leading;

    // case 4: tail<=6 and lead=pr_lead
    var case_4:vec2<u32>=vec2(0,3);// code 11
    case_4=pseudo_u64_shift(case_4,3);
    case_4=pseudo_u64_add(case_3,s.leading);
    case_4=pseudo_u64_shift(case_4,32 - s.leading);
    case_4=pseudo_u64_add(case_4,xorred%(32-s.leading));
    var case_4_bit_count:u32=2+3+32 - s.leading;

    var final_output=s.equal*case_1+ (trail_gt_6*not_equal)*case_2 +(trail_le_6*pr_lead_eq_lead)*case_3 +(trail_le_6*pr_lead_ne_lead)*case_4;
    var final_bit_count=s.equal*case_1_bit_count+ (trail_gt_6*not_equal)*case_2_bit_count +(trail_le_6*pr_lead_eq_lead)*case_3_bit_count +(trail_le_6*pr_lead_ne_lead)*case_4_bit_count;
    return Output(final_output,final_bit_count);
}

fn pseudo_u64_shift(output:vec2<u32>,number:u32)->vec2<u32>{
    // we are going to find what the first <number> bits were and shift them from the vec.y to vec.x
    var ones:u32=1<<number;

    var reverse_ones:u32=reverseBits(ones - 1);

    var first_number_bits:u32=reverseBits(ouput.y&&reverse_ones);

    ouput.x = output.x << number;
    ouput.x += first_number_bits;
    ouput.y = output.y<<number;

    return first_number_bits;
}


fn pseudo_u64_add(output:vec2<u32>,number:u32)->vec2<u32>{
    // check if adding <number> causes an overflow
    var max_u32:u32=0xffffffffu;
    var isOverflow:u32=u32(ouput.y>=(max_u32-number));
    var isNotOverflow:u32=u32(abs(1-isOverflow));
    var diff:u32=max_u32-output.y;

    ouput.x+=1*isOverflow;
    ouput.y=isOverflow*(number-diff) + isNotOverflow*(ouput.y+number);
    return ouput;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var index_prev=max(global_id.x - 1,u32(0));
    out[global_id.x] = compress(in[global_id.x],s_store[global_id.x],in[index_prev],s_store[index_prev]);
}