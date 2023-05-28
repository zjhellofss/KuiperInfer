
#include "utils/gpu_utils.cuh"
#include<stdio.h>
namespace kuiper_infer {

__global__ void rand_array_kernel(int n, float* array) {
  CUDA_KERNEL_LOOP(index, n) {
    array[index] = (index * 1.0f);
    // 生成随机数
    // printf("%d\n",index);
  }
}
void rand_array(int n, float* array) {
  rand_array_kernel<<<KUIPER_GET_BLOCKS(n), KUIPER_CUDA_NUM_THREADS>>>(n,
                                                                       array);
}

__global__ void element_wise_fill_kernel(int n, float* input, float value) {
  CUDA_KERNEL_LOOP(index, n) { input[index] = value; }
}

void element_wise_fill(int n, float* input, float value) {
  int num_blocks = KUIPER_GET_BLOCKS(n);
  element_wise_fill_kernel<<<num_blocks, KUIPER_CUDA_NUM_THREADS>>>(n, input,
                                                                    value);
}

__global__ void element_wise_add_kernel(int n, float* output,
                                        const float* input1,
                                        const float* input2) {
  CUDA_KERNEL_LOOP(index, n) { output[index] = input1[index] + input2[index]; }
}

void element_wise_add(int n, float* output, const float* input1,
                      const float* input2) {
  element_wise_add_kernel<<<KUIPER_GET_BLOCKS(n), KUIPER_CUDA_NUM_THREADS>>>(
      n, output, input1, input2);
}

}  // namespace kuiper_infer