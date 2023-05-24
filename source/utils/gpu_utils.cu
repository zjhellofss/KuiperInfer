
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

__global__ void element_wise_fill_kernel(int n, float* input, float value) {
  CUDA_KERNEL_LOOP(index, n) { input[index] = value; }
}

void element_wise_fill(int n, float* input, float value) {
  int num_blocks = KUIPER_GET_BLOCKS(n);
  element_wise_fill_kernel<<<num_blocks, KUIPER_CUDA_NUM_THREADS>>>(n, input,value);
}


void element_wise_add(int n, const float* input1, const float* input2,
                            float* output);

void element_wise_add(int n, const float* input1, const float* input2,
                            float* output){

}


}  // namespace kuiper_infer