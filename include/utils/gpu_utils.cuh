
#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

#define CUDA_CHECK(condition)                                         \
  /* Code block avoids redefinition of cudaError_t error */           \
  do {                                                                \
    cudaError_t error = condition;                                    \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition)                        \
  do {                                                 \
    cublasStatus_t status = condition;                 \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS)            \
        << " " << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition)                        \
  do {                                                 \
    curandStatus_t status = condition;                 \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS)            \
        << " " << caffe::curandGetErrorString(status); \
  } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// CUDA: check for error after kernel execution and exit loudly if there is one.
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

namespace kuiper_infer {

// CUDA: library error reporting.
const char* cublasGetErrorString(cublasStatus_t error);
const char* curandGetErrorString(curandStatus_t error);

// CUDA: use 512 threads per block
const int KUIPER_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int KUIPER_GET_BLOCKS(const int N) {
  return (N + KUIPER_CUDA_NUM_THREADS - 1) / KUIPER_CUDA_NUM_THREADS;
}



void element_wise_fill(int n, float* input, float value);
void element_wise_add(int n, const float* input1, const float* input2,
                            float* output);

}  // namespace kuiper_infer
