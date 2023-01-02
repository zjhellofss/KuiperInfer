#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK (512)

__global__ void im2col
(
    float *in, 

    int c_idx,
    int h_size,
    int h_size_w,
    int h_size_h,
    int w_size,

    int h_size_pad,
    int w_size_pad,
    
    int h,
    int w,
    int k,
    int s,

    float *out
    ){

        int bx = blockIdx.x;
        int tx = threadIdx.x;
        // printf("bx = %d\n", bx);
        int idx_col = bx * blockDim.x + tx;

        if(idx_col < h_size){
            float *out_c = out + c_idx * k * k + idx_col * w_size_pad;

            int idx_img = c_idx * h * w + 
                            idx_col / h_size_w * h * s + 
                            idx_col % h_size_w * s;
            
            float* in_c = in + idx_img;
            
            for(int i = 0; i < k; ++i){
                for (int j = 0; j < k; ++j){
                    out_c[i * k + j] = in_c[i * w + j];
                }
            }
        }
}

__global__ void kernel2col(
    float *in,
    int out_c,
    int in_c,
    int out_c_pad,
    int in_c_pad,
    int k,
    float *out
){
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int in_idx = bx * in_c * k * k + tx * k * k;
    int out_idx = bx * k * k * out_c_pad + tx % out_c;

    float *out_ = out + out_idx;
    float *in_ = in + in_idx;

    for(int i = 0; i < k * k; ++i){
        out_[out_c_pad * i] = in_[i];
    }

}


void launch_im2col(

    float *in, 

    int c,
    int h_size,
    int h_size_w,
    int h_size_h,
    int w_size,

    int h_size_pad,
    int w_size_pad,
    
    int h,
    int w,
    int k,
    int s,

    float *out){

        int threads = THREADS_PER_BLOCK;
        int grid = (h_size + threads - 1) / threads;
        printf("grid = %d\tthreads = %d\n", grid, threads);

        for (int i = 0; i < c; ++i){
            im2col<<<grid, threads>>>(
                in, 
                i,
                h_size,
                h_size_w,
                h_size_h,
                w_size,
                h_size_pad,
                w_size_pad,
                h,
                w,
                k,
                s,
                out);
            cudaDeviceSynchronize();
        }

}


void launch_kernel2col(
    float *in,
    int out_c,
    int in_c,
    int out_c_pad,
    int in_c_pad,
    int k,
    float *out
){
    int threads = out_c;
    int grid = in_c;

    kernel2col<<<grid, threads>>>(
        in,
        out_c,
        in_c,
        out_c_pad,
        in_c_pad,
        k,
        out);

}







