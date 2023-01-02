
// https://github.com/tpoisonooo/how-to-optimize-gemm.git

#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#define SMEM_LDA (128)
#define SMEM_LDB (128)

__global__ __launch_bounds__ (256, 2) 
void sgemm_128x128x8(int m, int n, int k, const float *a, const float *b, float *c)
{
    __shared__ __align__(16 * 1024) char smem[24 * 1024];
    float *ashare = reinterpret_cast<float*>(smem);
    float *bshare = reinterpret_cast<float*>(smem + 16 * 1024);
    float sum[8][8] = {0};
    float panelA[8] = {0}, panelB[8] = {0};

    int from_a = (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8;
    int from_b = blockIdx.x * 128 + threadIdx.x / 32 * n + threadIdx.x % 32;

    for(int loop = 0; loop < k; loop += 8){

        int to_a = threadIdx.x % 8 * SMEM_LDA + threadIdx.x / 8 * 4;
        for(int i = 0; i < 4; ++i){
            ashare[to_a + i] = a[from_a + i * k];
        }

        int to_b = threadIdx.x / 32 * SMEM_LDB + threadIdx.x % 32;
        for(int i = 0; i < 4; ++i){
            bshare[to_b + i * 32] = b[from_b + i * 32];
        }
        __syncthreads();

        from_a += 8;
        from_b += 8 * n;

        int aidx0 = threadIdx.x / 16 * 4;
        int bidx0 = threadIdx.x % 16 * 4;

        for(int subk = 0; subk < 8; ++subk){
            float *ptrA = ashare + aidx0 + subk * SMEM_LDA;
            for(int i = 0; i < 4; ++i){
                panelA[i] = ptrA[i];
                panelA[i + 4] = ptrA[i + 64]; 
            }

            float *ptrB = bshare + bidx0 + subk * SMEM_LDB;
            for(int i = 0; i < 4; ++i){
                panelB[i] = ptrB[i];
                panelB[i + 4] = ptrB[i + 64];
            }

            for(int i = 0; i < 8; ++i){
                for(int j = 0; j < 8; ++j){
                    sum[i][j] += panelA[i] * panelB[j];
                }
            }
        }
        __syncthreads();
    }

    int write_offset = (blockIdx.y * 128 + threadIdx.x / 16 * 4) * n +
                        blockIdx.x * 128 + threadIdx.x % 16 * 4;
    for(int i = 0; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            c[write_offset + i * n + j] = sum[i][j];
            c[write_offset + i * n + j + 64] = sum[i][j+4];
            c[write_offset + (i + 64) * n + j] =sum[i+4][j]; 
            c[write_offset + (i + 64) * n + j + 64] =sum[i+4][j+4]; 
        }
    }
}

template <int BLOCK>
__global__ void sgemm(int m, int n, int k, float *a, float *b, float *c) {
  // blockIdx control subpanel matrix

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  //printf("tx = %d, ty = %d, bx = %d, by = %d, BLOCK = %d, k = %d, by * BLOCK * k = %d\n", tx, ty, bx, by, BLOCK, k, by * BLOCK * k);
  float *begin_a = a + by * blockDim.y * k;
  float *begin_b = b + bx * blockDim.x;
  float *end_a = begin_a + k;

  float sum = 0.f;
  for (float *a_ptr = begin_a, *b_ptr = begin_b; a_ptr < end_a;
       a_ptr += BLOCK, b_ptr += BLOCK * n) {
    __shared__ float ashare[BLOCK][BLOCK];
    __shared__ float bshare[BLOCK][BLOCK];

    ashare[ty][tx] = a_ptr[ty * k + tx];
    bshare[ty][tx] = b_ptr[ty * n + tx];
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BLOCK; ++kk) {
      sum += ashare[ty][kk] * bshare[kk][tx];
    }
    __syncthreads();
  }

  c[(BLOCK * by + ty) * n + BLOCK * bx + tx] = sum;
}

template <int BLOCK>
__global__ void sgemm_naive(int m, int n, int k, float *a, float *b, float *c) {
  int _m = blockIdx.x * BLOCK + threadIdx.x;
  int _n = blockIdx.y * BLOCK + threadIdx.y;
  if (_m < m and _n < n) {
    float sum = 0.f;
    for (int i = 0; i < k; ++i) {
      sum += a[_m * k + i] * b[i * n + _n];
    }
    c[_m * n + _n] = sum;
  }
}


void launch_sgemm(int m, int n, int k, float *a, float *b, float *c){

    if (m >= 32 && n >= 32 && k >=8){
        constexpr int block = 128;
        dim3 grid((m + block -1) / block, (n + block -1) / block);
        sgemm_128x128x8<<<grid, 256>>>(m, n, k, a, b, c);
    }
    else if (m >= 32 && n >= 32 && k >= 1){
        constexpr int BLOCK = 32;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((m + BLOCK -1) / BLOCK, (n + BLOCK -1) / BLOCK);
        sgemm<BLOCK><<<grid, block>>>(m, n, k, a, b, c);
    }
    else {
        constexpr int BLOCK = 16;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((m + BLOCK -1) / BLOCK, (n + BLOCK -1) / BLOCK);
        sgemm_naive<BLOCK><<<grid, block>>>(m, n, k, a, b, c);
    }
}




















