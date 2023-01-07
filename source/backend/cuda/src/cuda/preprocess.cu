#include <cuda_runtime.h>
#include <stdio.h>

constexpr unsigned int kThreadsPerBlock = 256;
constexpr float kDefaultFillValue = 114.f/255.f;

__global__ void setValueKernel(float *values, int numValue, float value){

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if( tid < numValue ){
        values[tid] = value;
    }
}

template<typename T>
__device__ inline T clipValue(T val, T floor, T ceil){
    if (val < floor){
        val = floor;
    }
    if (val > ceil){
        val = ceil;
    }
    return val;
}

__global__ void resizKernel(
    float *mean, float *std,
    const char *in_image, int in_height, int in_width,
    float *out_image, int out_height, int out_width,
    int letter_height, int letter_width, float ratio,
    int offset_row, int offset_col){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < out_height * out_width){
        int out_row = tid / out_width;
        int out_col = tid % out_width;

        int out_start = out_col + out_row * out_width;
        // printf("out_start = %d\n", out_start);
        float *out_r = out_image + out_start;
        float *out_g = out_image + out_start + out_height * out_width;
        float *out_b = out_image + out_start + out_height * out_width * 2;

        if (out_row < offset_row || out_row >= letter_height + offset_row ||
            out_col < offset_col || out_col >= letter_width + offset_col){

            *out_r = (kDefaultFillValue - mean[0]) /  std[0];
            *out_g = (kDefaultFillValue - mean[1]) /  std[1];
            *out_b = (kDefaultFillValue - mean[2]) /  std[1];;
        }else{
            // printf("tid = %d\tsize = %d\n", tid, out_height * out_width);
            float in_row_f = (float)(out_row - offset_row) / ratio;
            float in_col_f = (float)(out_col - offset_col) / ratio;

            int in_row = (int)in_row_f;
            int in_col = (int)in_col_f;
            in_row = clipValue<int>(in_row, 0, in_height - 1);
            in_col = clipValue<int>(in_col, 0, in_width - 1);

            int x_0_y_0 = in_col + in_row * in_width;
            int x_1_y_0 = (in_col + 1) + in_row * in_width;
            int x_0_y_1 = in_col + (in_row + 1) * in_width;
            int x_1_y_1 = (in_col + 1) + (in_row + 1) * in_width;

            float W_x_0 = floor(in_col_f) - in_col_f;
            float W_x_1 = 1.f - W_x_0;
            float W_y_0 = floor(in_row_f) - in_row_f;
            float W_y_1 = 1.f - W_y_0;
            // printf("in_image[x_0_y_0 * 3 + 2] = %d\n", in_image[x_0_y_0 * 3 + 2]);

            *out_r = (((float)in_image[x_0_y_0 * 3 + 2] * W_x_0 * W_y_0 +
                        (float)in_image[x_1_y_0 * 3 + 2] * W_x_1 * W_y_0 +
                        (float)in_image[x_0_y_1 * 3 + 2] * W_x_0 * W_y_1 +
                        (float)in_image[x_1_y_1 * 3 + 2] * W_x_1 * W_y_1) / 255.f - mean[0]) / std[0];

            *out_g = (((float)in_image[x_0_y_0 * 3 + 1] * W_x_0 * W_y_0 +
                        (float)in_image[x_1_y_0 * 3 + 1] * W_x_1 * W_y_0 +
                        (float)in_image[x_0_y_1 * 3 + 1] * W_x_0 * W_y_1 +
                        (float)in_image[x_1_y_1 * 3 + 1] * W_x_1 * W_y_1) / 255.f - mean[1]) / std[1];

            *out_b = (((float)in_image[x_0_y_0 * 3] * W_x_0 * W_y_0 +
                        (float)in_image[x_1_y_0 * 3] * W_x_1 * W_y_0 +
                        (float)in_image[x_0_y_1 * 3] * W_x_0 * W_y_1 +
                        (float)in_image[x_1_y_1 * 3] * W_x_1 * W_y_1) / 255.f - mean[2]) / std[2];
            // printf("*out_r = %f,\t *out_g = %f,\t *out_b = %f\n", *out_r, *out_g, *out_b);
            // printf("In resize_kernel\n");
        }
    }
}

__global__ void resizKernel(
    const char *in_image, int in_height, int in_width,
    float *out_image, int out_height, int out_width,
    int letter_height, int letter_width, float ratio,
    int offset_row, int offset_col){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < out_height * out_width){
        int out_row = tid / out_width;
        int out_col = tid % out_width;

        int out_start = out_col + out_row * out_width;
        // printf("out_start = %d\n", out_start);
        float *out_r = out_image + out_start;
        float *out_g = out_image + out_start + out_height * out_width;
        float *out_b = out_image + out_start + out_height * out_width * 2;

        if (out_row < offset_row || out_row >= letter_height + offset_row ||
            out_col < offset_col || out_col >= letter_width + offset_col){

            *out_r = kDefaultFillValue;
            *out_g = kDefaultFillValue;
            *out_b = kDefaultFillValue;;
        }else{
            // printf("tid = %d\tsize = %d\n", tid, out_height * out_width);
            float in_row_f = (float)(out_row - offset_row) / ratio;
            float in_col_f = (float)(out_col - offset_col) / ratio;

            int in_row = (int)in_row_f;
            int in_col = (int)in_col_f;
            in_row = clipValue<int>(in_row, 0, in_height - 1);
            in_col = clipValue<int>(in_col, 0, in_width - 1);

            int x_0_y_0 = in_col + in_row * in_width;
            int x_1_y_0 = (in_col + 1) + in_row * in_width;
            int x_0_y_1 = in_col + (in_row + 1) * in_width;
            int x_1_y_1 = (in_col + 1) + (in_row + 1) * in_width;

            float W_x_0 = floor(in_col_f) - in_col_f;
            float W_x_1 = 1.f - W_x_0;
            float W_y_0 = floor(in_row_f) - in_row_f;
            float W_y_1 = 1.f - W_y_0;
            // printf("in_image[x_0_y_0 * 3 + 2] = %d\n", in_image[x_0_y_0 * 3 + 2]);

            *out_r = ((float)in_image[x_0_y_0 * 3 + 2] * W_x_0 * W_y_0 +
                        (float)in_image[x_1_y_0 * 3 + 2] * W_x_1 * W_y_0 +
                        (float)in_image[x_0_y_1 * 3 + 2] * W_x_0 * W_y_1 +
                        (float)in_image[x_1_y_1 * 3 + 2] * W_x_1 * W_y_1) / 255.f;

            *out_g = ((float)in_image[x_0_y_0 * 3 + 1] * W_x_0 * W_y_0 +
                        (float)in_image[x_1_y_0 * 3 + 1] * W_x_1 * W_y_0 +
                        (float)in_image[x_0_y_1 * 3 + 1] * W_x_0 * W_y_1 +
                        (float)in_image[x_1_y_1 * 3 + 1] * W_x_1 * W_y_1) / 255.f;

            *out_b = ((float)in_image[x_0_y_0 * 3] * W_x_0 * W_y_0 +
                        (float)in_image[x_1_y_0 * 3] * W_x_1 * W_y_0 +
                        (float)in_image[x_0_y_1 * 3] * W_x_0 * W_y_1 +
                        (float)in_image[x_1_y_1 * 3] * W_x_1 * W_y_1) / 255.f;
        }
    }
}

void launchResizeKernel(
    const char *in_image, int in_height, int in_width,
    float *out_image, int out_height, int out_width,
    int letter_height, int letter_width, float ratio,
    int offset_row, int offset_col){

    int blocks = (out_height * out_width + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dim3 grid(blocks);

    resizKernel<<<grid, kThreadsPerBlock>>>(
        in_image, in_height, in_width, 
        out_image, out_height, out_width,
        letter_height, letter_width, ratio,
        offset_row, offset_col);
    cudaDeviceSynchronize();

}

void launchResizeKernel(
    float *mean, float *std,
    const char *in_image, int in_height, int in_width,
    float *out_image, int out_height, int out_width,
    int letter_height, int letter_width, float ratio,
    int offset_row, int offset_col){

    int blocks = (out_height * out_width + kThreadsPerBlock - 1) / kThreadsPerBlock;
    dim3 grid(blocks);

    resizKernel<<<grid, kThreadsPerBlock>>>(mean, std,
        in_image, in_height, in_width, 
        out_image, out_height, out_width,
        letter_height, letter_width, ratio,
        offset_row, offset_col);
    cudaDeviceSynchronize();

}

















