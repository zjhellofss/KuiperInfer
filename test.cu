#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

// CUDA内核函数，用于生成随机数

void fuck() { int a = 1; }
__global__ void randomKernel(float* randomArray, int arraySize) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < arraySize) {
    printf("32\n");
    curandState_t state;
    curand_init(0, tid, 0, &state);  // 使用0作为种子值，tid是线程ID
    fuck();
    // 生成随机数
    randomArray[tid] = curand_uniform(&state);
  }
}

int main() {
  int arraySize = 1000;
  float randomArray[arraySize];

  // 分配和初始化设备数组
  float* devRandomArray;
  cudaMalloc((void**)&devRandomArray, arraySize * sizeof(float));
  cudaMemcpy(devRandomArray, randomArray, arraySize * sizeof(float),
             cudaMemcpyHostToDevice);

  // 启动内核函数
  int blockSize = 512;
  int gridSize = (arraySize + blockSize - 1) / blockSize;
  randomKernel<<<gridSize, blockSize>>>(devRandomArray, arraySize);

  // 将结果从设备复制回主机
  cudaMemcpy(randomArray, devRandomArray, arraySize * sizeof(float),
             cudaMemcpyDeviceToHost);

  // 打印随机数

  // 清理资源
  cudaFree(devRandomArray);

  return 0;
}
