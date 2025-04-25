#version 450
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require


layout(local_size_x = @@workgroup_size) in;

struct S {
    int leading;
    int trailing;
    uint equal;
};

layout(set = 0, binding = 0) buffer SStore {
    S s_store[];
};

layout(set = 0, binding = 1) buffer InBuffer {
    double in_data[];
};

S calculate_s(uint id, double v_prev, double v) {
    uint64_t v_prev_u64 = uint64_t(v_prev);
    uint64_t v_u64 = uint64_t(v);
    uint64_t i = v_prev_u64 ^ v_u64;

    int leading = (id % 256 != 0) ? int(findMSB(i) ^ 63) : 0;
    int trailing = int(findLSB(i));
    uint equal = (i == 0UL) ? 1U : 0U;

    return S(leading, trailing, equal);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    s_store[id + 1] = calculate_s(@@start_index+id, in_data[id], in_data[id + 1]);
}