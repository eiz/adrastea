#version 450
#extension GL_EXT_buffer_reference: enable
layout (buffer_reference) buffer PFloat32 {
    float data[];
};
layout (push_constant) uniform SquareArgs {
    PFloat32 in_buf;
    PFloat32 out_buf;
    uint height;
    uint width;
};
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint index = y * width + x;
    if (x < width && y < height) {
        out_buf.data[index] = in_buf.data[index] * in_buf.data[index];
    }
}
