extern "C" __global__ void square_fp32_16x16(float* output, float* input, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * width + x;
  if (x < width && y < height) {
    output[idx] = input[idx] * input[idx];
  }
}