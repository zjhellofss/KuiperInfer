// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
// Created by fss on 23-10-11.
//
#include "deconvolution.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {

void DeconvolutionLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << "The set weights function does not support this convolution type: "
             << int32_t(conv_type_);
}

void DeconvolutionLayer::set_weights(const std::vector<float>& weights) {
  const uint32_t kernel_count = this->weights_.size();

  CHECK_GT(kernel_count, 0);
  const uint32_t kernel_count_group = kernel_count / groups_;
  const uint32_t kernel_channel = this->weights_.at(0)->channels();
  uint32_t kernel_height = this->weights_.at(0)->rows();
  uint32_t kernel_width = this->weights_.at(0)->cols();
  if (dilation_h_ > 1) {
    kernel_height = (kernel_height + dilation_h_ - 1) / dilation_h_;
  }
  if (dilation_w_ > 1) {
    kernel_width = (kernel_width + dilation_w_ - 1) / dilation_w_;
  }

  CHECK_EQ(kernel_count * kernel_channel * kernel_width * kernel_height, weights.size());

  const uint32_t kernel_hw = kernel_height * kernel_width;
  const uint32_t kernel_nhw = kernel_count_group * kernel_hw;
  const uint32_t kernel_plane = kernel_channel * kernel_nhw;

  for (uint32_t group = 0; group < groups_; ++group) {
    // sub_weights表示一个group内所有卷积核的权重值
    std::vector<float> sub_weights(kernel_plane);
    std::copy(weights.data() + group * kernel_plane, weights.data() + (group + 1) * kernel_plane,
              sub_weights.begin());
    for (uint32_t kg = 0; kg < kernel_count_group; ++kg) {
      const uint32_t channel_offset = kg * kernel_hw;
      const uint32_t kernel_idx = group * kernel_count_group + kg;
      /*
       * 卷积核权重摆放的顺序是c n h w， 需要将它调整到n c h w
       * 其中n表示卷积核次序，kernel_idx = group * kernel_count_group + kg;
       * origin_pixel_idx = ic * kernel_nhw (nhw) + kg(n) * kernel_hw + ...
       */
      for (uint32_t ic = 0; ic < kernel_channel; ++ic) {
        const uint32_t kernel_offset = ic * kernel_nhw;
        arma::fmat& kernel_channel_mat = this->weights_.at(kernel_idx)->slice(ic);

        for (uint32_t kw = 0; kw < kernel_width; ++kw) {
          uint32_t kw_dilation = kw * dilation_w_;
          float* kernel_ptr = kernel_channel_mat.colptr(kw_dilation);
          for (uint32_t kh = 0; kh < kernel_height; ++kh) {
            uint32_t kh_dilation = kh * dilation_h_;
            *(kernel_ptr + kh_dilation) =
                sub_weights.at(kernel_offset + channel_offset + kh * kernel_width + kw);
          }
        }
      }
    }
  }
}

void DeconvolutionLayer::ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h,
                                       uint32_t kernel_w, uint32_t kernel_count_group,
                                       uint32_t input_h, uint32_t input_w,
                                       uint32_t channels_per_group, uint32_t output_h,
                                       uint32_t output_w, uint32_t group) const {
#pragma omp parallel for
  for (uint32_t k = 0; k < kernel_count_group; ++k) {
    const arma::fmat& gemm_result =
        DeconvGEMM(input, input_h, input_w, channels_per_group, group, k, kernel_count_group);
    DeconvCol2ImBias(gemm_result, output_tensor, input_h, input_w, group, k, kernel_count_group,
                     kernel_h, kernel_w, output_h, output_w);
  }
}

std::pair<uint32_t, uint32_t> DeconvolutionLayer::ComputeOutputSize(const uint32_t input_h,
                                                                    const uint32_t input_w,
                                                                    const uint32_t kernel_h,
                                                                    const uint32_t kernel_w) const {
  uint32_t output_h = 0;
  uint32_t output_w = 0;

  output_h = (input_h - 1) * stride_h_ + kernel_h + output_padding_h_;
  output_w = (input_w - 1) * stride_w_ + kernel_w + output_padding_w_;
  CHECK(output_h > 2 * padding_h_ && output_w > 2 * padding_w_);
  output_h -= 2 * padding_h_;
  output_w -= 2 * padding_w_;
  return {output_h, output_w};
}

arma::fmat DeconvolutionLayer::DeconvGEMM(const sftensor& input, uint32_t input_h, uint32_t input_w,
                                          uint32_t channels_per_group, uint32_t group,
                                          uint32_t kernel_index,
                                          uint32_t kernel_count_group) const {
  CHECK(input != nullptr && !input->empty());

  kernel_index = kernel_index + group * kernel_count_group;
  CHECK(kernel_index < this->weights_.size());

  sftensor group_kernel = this->weights_.at(kernel_index);
  CHECK(group_kernel != nullptr && !group_kernel->empty());

  uint32_t input_hw = input_h * input_w;
  uint32_t kernel_hw = group_kernel->rows() * group_kernel->cols();

  arma::fmat multi_input_channel(input->matrix_raw_ptr(group * channels_per_group), input_hw,
                                 channels_per_group, false, true);

  arma::fmat multi_kernel_channel(group_kernel->raw_ptr(), kernel_hw, channels_per_group, false,
                                  true);
  return multi_kernel_channel * (multi_input_channel.t());
}

void DeconvolutionLayer::DeconvCol2ImBias(const arma::fmat& gemm_result, sftensor output_tensor,
                                          uint32_t input_h, uint32_t input_w, uint32_t group,
                                          uint32_t kernel_index, uint32_t kernel_count_group,
                                          uint32_t kernel_h, uint32_t kernel_w, uint32_t output_h,
                                          uint32_t output_w) const {
  CHECK(!gemm_result.empty());
  CHECK(input_h > 0 && input_w > 0);
  CHECK(output_tensor != nullptr && !output_tensor->empty());

  arma::fmat output_padding(output_h + 2 * padding_h_, output_w + 2 * padding_w_);

  uint32_t slide_w = input_w;
  uint32_t slide_h = input_h;
  uint32_t slide_size = slide_h * slide_w;
  for (uint32_t index = 0; index < slide_size; ++index) {
    uint32_t x = index / slide_h;
    uint32_t y = index % slide_h;
    const uint32_t offset_x = x * stride_w_;
    const uint32_t offset_y = y * stride_h_;
    arma::fmat gemm_column(const_cast<float*>(gemm_result.colptr(index)), kernel_h, kernel_w, false,
                           true);

    uint32_t gemm_rows = gemm_column.n_rows;
    uint32_t gemm_cols = gemm_column.n_cols;

    for (uint32_t col = 0; col < gemm_cols; ++col) {
      float* column_ptr = gemm_column.colptr(col);
      float* output_ptr = output_padding.colptr(offset_x + col);
      for (uint32_t row = 0; row < gemm_rows; ++row) {
        *(output_ptr + offset_y + row) += *(column_ptr + row);
      }
    }
  }

  kernel_index = kernel_index + group * kernel_count_group;
  arma::fmat output(output_tensor->matrix_raw_ptr(kernel_index), output_h, output_w, false, true);

  output = output_padding.submat(padding_h_, padding_w_, output_h + padding_h_ - 1,
                                 output_w + padding_w_ - 1);

  return AddBias(output, kernel_index);
}

LayerRegistererWrapper kDeConvCreateInstance(BaseConvolutionLayer::CreateInstance,
                                             "nn.ConvTranspose2d");
}  // namespace kuiper_infer