#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <cuda_runtime.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


using namespace std;

#define CHECK(call)                                                                        \
{                                                                                          \
    const cudaError_t error = call;                                                        \
    if (error != cudaSuccess){                                                             \
        printf("Error: %s: %d, ", __FILE__, __LINE__);                                     \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));                \
        exit(1);                                                                           \
    }                                                                                      \
}                                                                                          \

int kernel_times(int w, int k, int s, int p);

void window_cal(const int w, const int k, const int s, const int p, int win,
                std::vector<int>* conv_idx,
                std::vector<int>* tile_idx,
                std::vector<int>* ac_before_tile,
                std::vector<int>* tile_size,
                std::vector<int>* in_win);
void print_var(std::vector<int>* var, string *s);
void print_var(std::vector<float>* var, string *s);
void init_in(std::vector<float>* img, int H, int W);
void init_out(std::vector<float>* img, int H, int W);
void init_kernel(std::vector<float>* img, int out_c, int in_c, int k);
int pad(int w, int v);
void init(float* df, int h, int w);



extern void launch_im2col(

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

    float *out);

extern void launch_kernel2col(
    float *in,
    int out_c,
    int in_c,
    int out_c_pad,
    int in_c_pad,
    int k,
    float *out
);

extern void launch_sgemm(int m, int n, int k, float *a, float *b, float *c);

void launchResizeKernel(
    float *mean, float *std,
    const char *in_image, int in_height, int in_width,
    float *out_image, int out_height, int out_width,
    int letter_height, int letter_width, float ratio,
    int offset_row, int offset_col);

void launchResizeKernel(
    const char *in_image, int in_height, int in_width,
    float *out_image, int out_height, int out_width,
    int letter_height, int letter_width, float ratio,
    int offset_row, int offset_col);

int main(){

        // // ! img
        // int c = 1;
        // int h = 128;
        // int w = 128;
        // // int h = 16;
        // // int w = 16;
        // int k = 3;
        // int p = 0;
        // int s = 1;

        // const int pad_h = 128;
        // const int pad_w = 8;

        // int h_size_w = kernel_times(w, k, s, p);
        // int h_size_h = kernel_times(h, k, s, p);
        // int h_size = h_size_w * h_size_h;
        // int w_size = k * k * c;

        // int h_size_pad = pad(h_size, pad_h);
        // int w_size_pad = pad(w_size, pad_w);



        // std::vector<float> h_in;
        // init_in(&h_in, h, w);
        // float *d_in;
        // cudaMalloc(&d_in, h_in.size() * sizeof(float));
        // cudaMemcpy(d_in, &h_in[0], h_in.size() * sizeof(float), cudaMemcpyHostToDevice);

        // std::vector<float> h_out;
        // std::cout<<"h_size = "<<h_size<<" w_size = "<<w_size<<std::endl;
        // init_out(&h_out, h_size_pad, w_size_pad);
        // float *d_out;
        // cudaMalloc(&d_out, h_out.size() * sizeof(float));
        // cudaMemcpy(d_out, &h_out[0], h_out.size() * sizeof(float), cudaMemcpyHostToDevice);

        // launch_im2col(
        //     d_in, 

        //     c,
        //     h_size,
        //     h_size_w,
        //     h_size_h,
        //     w_size,

        //     h_size_pad,
        //     w_size_pad,

        //     h,
        //     w,
        //     k,
        //     s,

        //     d_out);

        // cudaMemcpy(&h_out[0], d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // std::cout<<"h_out.size() = "<<h_out.size()<<"\th_size = "<<h_size<<" h_size_h = "<<h_size_h<<" h_size_w = "<<h_size_w<<std::endl;
        // ofstream ofs;
        // ofs.open("col.txt", ios::out);
        // // for(int i = 0; i < h_out.size(); ++i){
        // //     ofs<<h_out.at(i*h_size_w+j)<<" ";
        // // }
        // for(int i = 0; i < h_size_pad; ++i){
        //     for(int j = 0; j < w_size_pad; ++j){
        //         ofs<<h_out.at(i*w_size_pad+j)<<" ";
        //     }
        //     ofs<<"\n";
        //     // std::cout<<"\n";
        // }

        // cudaFree(d_in);
        // cudaFree(d_out);

        // // ! kernel
        // int out_c = 2;
        // int in_c = 2;
        // int out_c_pad = out_c;
        // int in_c_pad = in_c;
        // k = 3;
        // std::vector<float> h_kernel_in;
        // init_kernel(&h_kernel_in, out_c, in_c, k);
        // float *d_kernel_in;
        // cudaMalloc(&d_kernel_in, h_kernel_in.size() * sizeof(float));
        // cudaMemcpy(d_kernel_in, &h_kernel_in[0], h_kernel_in.size() * sizeof(float), cudaMemcpyHostToDevice);

        // std::vector<float> h_kernel_out;
        // init_kernel(&h_kernel_in, out_c, in_c, k);
        // init_kernel(&h_kernel_out, out_c_pad, in_c_pad, k);
        // float *d_kernel_out;
        // cudaMalloc(&d_kernel_out, h_kernel_out.size() * sizeof(float));
        // cudaMemcpy(d_kernel_out, &h_kernel_out[0], h_kernel_out.size() * sizeof(float), cudaMemcpyHostToDevice);
        // launch_kernel2col(d_kernel_in, out_c, in_c, out_c_pad, in_c_pad, k, d_kernel_out);
        // cudaMemcpy(&h_kernel_out[0], d_kernel_out, h_kernel_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // ofstream ofs_kernel_in;
        // ofs_kernel_in.open("kernelIn.txt", ios::out);
        // for(int i = 0; i < out_c; ++i){
        //     for(int j = 0; j < in_c * k * k; ++j){
        //         ofs_kernel_in<<h_kernel_in.at(i*in_c * k * k+j)<<" ";
        //     }
        //     ofs_kernel_in<<"\n";
        // }

        // ofstream ofs_kernel;
        // ofs_kernel.open("kernelOut.txt", ios::out);
        // for(int i = 0; i < in_c * k * k; ++i){
        //     for(int j = 0; j < out_c; ++j){
        //         ofs_kernel<<h_kernel_out.at(i*out_c_pad+j)<<" ";
        //     }
        //     ofs_kernel<<"\n";
        // }

        // // ! gemm
        // constexpr int block = 128;
        // int m = 16;
        // int n = 16;
        // k = 32;

        // float* h_A = (float*)malloc(m * k * sizeof(float));
        // float* h_B = (float*)malloc(k * n * sizeof(float));
        // float* h_C = (float*)malloc(m * n * sizeof(float));

        // init(h_A, m, k);
        // init(h_B, k, n);

        // float* d_A;
        // float* d_B;
        // float* d_C;

        // cudaMalloc(&d_A, m * k * sizeof(float));
        // cudaMalloc(&d_B, k * n * sizeof(float));
        // cudaMalloc(&d_C, m * n * sizeof(float));

        // cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
        // // cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);
        // dim3 grid((m + block -1) / block, (n + block -1) / block);
        // printf("grid.x = %d\tgrid.y = %d\n", (m + block -1) / block, (n + block -1) / block);

        // launch_sgemm(m, n, k, d_A, d_B, d_C);

        // cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

        // ofstream ofs_gemm;
        // ofs_gemm.open("gemm.txt", ios::out);
        // // for(int i = 0; i < h_out.size(); ++i){
        // //     ofs<<h_out.at(i*h_size_w+j)<<" ";
        // // }
        // for(int i = 0; i < m; ++i){
        //     for(int j = 0; j < n; ++j){
        //         ofs_gemm<<h_C[i * n + j]<<" ";
        //     }
        //     ofs_gemm<<"\n";
        //     ofs_gemm<<"\n";
        //     // std::cout<<"\n";
        // }

        // !preprocess
        string path = "zidane.jpg";
        cv::Mat img = cv::imread(path);
        // for(int i = 0; i < 100; ++i){
        //     printf("img.data[i] = %d\n", img.data[i]);
        // }

        int in_height = img.rows;
        int in_width = img.cols;
        int out_length = 640;
        int pad_size = 32;

        float ratio = std::min((float)out_length / (float)in_height, (float)out_length / (float)in_width);
        int letter_height = (int)(ratio * (float)in_height);
        int letter_width = (int)(ratio * (float)in_width);

        int out_height = pad(letter_height, pad_size);
        int out_width = pad(letter_width, pad_size);


        int offset_row = (out_height - letter_height) / 2;
        int offset_col = (out_width - letter_width) / 2;

        char *d_in_image;
        cudaMalloc(&d_in_image, in_height * in_width * 3 * sizeof(uint8_t));
        cudaMemcpy(d_in_image, img.data, in_height * in_width * 3 * sizeof(char), cudaMemcpyHostToDevice);

        float *d_out_image;
        cudaMalloc(&d_out_image, out_height * out_width * 3 * sizeof(float));

        float h_mean[3] = {0.485, 0.456, 0.406};
        float h_std[3] = {0.229, 0.224, 0.225};
        float *d_mean;
        float *d_std;
        cudaMalloc(&d_mean, 3 * sizeof(float));
        cudaMalloc(&d_std, 3 * sizeof(float));
        cudaMemcpy(d_mean, h_mean, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_std, h_std, 3 * sizeof(float), cudaMemcpyHostToDevice);

        launchResizeKernel(
            d_mean, d_std,
            d_in_image, in_height, in_width,
            d_out_image, out_height, out_width,
            letter_height, letter_width, ratio,
            offset_row, offset_col);
        
        float *h_out_image = (float*)malloc(out_height * out_width * 3 * sizeof(float));
        cudaMemcpy(h_out_image, d_out_image, out_height * out_width * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        ofstream pre_ofs;
        pre_ofs.open("preprocess.txt", ios::out);
        std::cout<<"in_height = "<<in_height<<" in_width = "<<in_width<<std::endl;
        std::cout<<"ratio = "<<ratio<<" letter_height = "<<letter_height<<" letter_width = "<<letter_width<<std::endl;
        std::cout<<"out_height = "<<out_height<<" out_width = "<<out_width<<std::endl;
        for(int i = 0; i < out_height * out_width * 3; ++i){
            pre_ofs<<h_out_image[i]<<"\n";
        }

        return 0;


}

int kernel_times(int w, int k, int s, int p){
    int res = (w - k + 2 * p) / s + 1;
    return res;

}

void print_var(std::vector<int>* var, string *s){
    std::cout<<std::endl;
    std::cout<<*s<<" = "<<std::endl;
    for(int i = 0; i < var->size(); ++i){
        std::cout<<var->at(i)<<"\t";
    }
    std::cout<<std::endl;
}

void print_var(std::vector<float>* var, string *s){
    std::cout<<std::endl;
    std::cout<<*s<<" = "<<std::endl;
    for(int i = 0; i < var->size(); ++i){
        std::cout<<var->at(i)<<"\t";
    }
    std::cout<<std::endl;
}

void window_cal(const int w, const int k, const int s, const int p, int win,
                std::vector<int>* conv_idx,
                std::vector<int>* tile_idx,
                std::vector<int>* ac_before_tile,
                std::vector<int>* tile_size,
                std::vector<int>* in_win
){
    int conv_cnts =kernel_times(w, k, s, p); 
    int cnt = 0;
    for(int i = 0; i < conv_cnts; ++i){
        conv_idx->push_back(cnt);
        cnt += s;
    }

    int flag = 0;
    int win_change_flag = 1;

    for(int i = 0; i < conv_cnts; ++i){
        if (w - conv_idx->at(i) < win){
            win_change_flag = 0;
            win = w - conv_idx->at(i);
            tile_idx->push_back(flag);
            ac_before_tile->push_back(i);
            int tile_size_one = kernel_times(win, k, s, p);
            tile_size->push_back(tile_size_one);
            in_win->push_back(win);
            break;
        }
        if (0 == i){
            flag = conv_idx->at(i);
            tile_idx->push_back(flag);
            ac_before_tile->push_back(i);
            int tile_size_one = kernel_times(win, k, s, p);
            tile_size->push_back(tile_size_one);
            in_win->push_back(win);
        }
        else if(conv_idx->at(i) - flag + k <= win){

        }
        else if((conv_idx->at(i) - flag < win) && (conv_idx->at(i) - flag + k > win)){
            flag = conv_idx->at(i);
            tile_idx->push_back(flag);
            ac_before_tile->push_back(i);
            tile_size->push_back(kernel_times(win, k, s, p));
            in_win->push_back(win);
        }
        else if(conv_idx->at(i) - flag >= win){
            flag = conv_idx->at(i);
            tile_idx->push_back(flag);
            ac_before_tile->push_back(i);
            tile_size->push_back(kernel_times(win, k, s, p));
            in_win->push_back(win);
        }
    }

}

void init_in(std::vector<float>* img, int H, int W){
    for(int i = 0; i < H; ++i){
        for(int j = 0; j < W; ++j){
            img->push_back(i * W + j);
            // std::cout<<"in = "<<img->at(i * W + j)<<std::endl;
        }
    }
}

void init_out(std::vector<float>* img, int H, int W){
    for(int i = 0; i < H; ++i){
        for(int j = 0; j < W; ++j){
            img->push_back(0);
        }
    }
}

int pad(int w, int v){
    return (w + v - 1) &(-v);
}

void init_kernel(std::vector<float>* img, int out_c, int in_c, int k){

    int out_size = in_c * k * k;
    for(int i = 0; i < out_size; ++i){
        for(int j = 0; j < k * k; ++j){
            for(int idx = 0; idx < k * k; ++idx){
                img->push_back(idx);
            }
        }
    }

}

void init(float* df, int h, int w){
    for(int i = 0; i < h; ++i){
        for(int j = 0; j < w; ++j){
            df[i * w + j] = 0.01 * (i * w + j);
            // df[i * w + j] = 1;
        }

    }
}
