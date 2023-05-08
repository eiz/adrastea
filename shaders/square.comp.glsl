#version 450
layout (binding = 0) readonly buffer InputBuffer {
    float in_buf[];
};
layout (binding = 1) writeonly buffer OutputBuffer {
    float out_buf[];
};
layout (push_constant) uniform WidthHeight {
    uint height;
    uint width;
};
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint index = y * width + x;
    if (x < width && y < height) {
        out_buf[index] = in_buf[index] * in_buf[index];
    }
}
