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

// Created by fss on 22-11-13.

#ifndef KUIPER_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#define KUIPER_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
#include "base_convolution.hpp"
#include "layer/abstract/param_layer.hpp"

namespace kuiper_infer {

class ConvolutionLayer : public BaseConvolutionLayer {
 public:
  explicit ConvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h,
                            uint32_t kernel_w, uint32_t padding_h, uint32_t padding_w,
                            uint32_t stride_h, uint32_t stride_w, uint32_t groups,
                            bool use_bias = true, uint32_t output_padding_h = 0,
                            uint32_t output_padding_w = 0, uint32_t dilation_h = 1,
                            uint32_t dilation_w = 1)
      : BaseConvolutionLayer(ConvType::kOpConv, output_channel, in_channel, kernel_h, kernel_w,
                             padding_h, padding_w, stride_h, stride_w, groups, use_bias,
                             output_padding_h, output_padding_w, dilation_h, dilation_w) {}

 private:
  bool Is1x1KernelNoPadding(uint32_t kernel_h, uint32_t kernel_w) const;

  void InitIm2ColWeight() override;

  void ComputeOutput(sftensor input, sftensor output_tensor, uint32_t kernel_h, uint32_t kernel_w,
                     uint32_t kernel_count_group, uint32_t input_h, uint32_t input_w,
                     uint32_t channels_per_group, uint32_t output_h, uint32_t output_w,
                     uint32_t group) const override;

  std::pair<uint32_t, uint32_t> ComputeOutputSize(uint32_t input_h, uint32_t input_w,
                                                  uint32_t kernel_h,
                                                  uint32_t kernel_w) const override;

  void ConvGEMMBias(const arma::fmat& input_matrix, sftensor output_tensor, uint32_t group,
                    uint32_t kernel_index, uint32_t kernel_count_group, uint32_t output_h,
                    uint32_t output_w, bool is_1x1conv_nopadding) const;

  [[nodiscard]] arma::fmat ConvIm2Col(sftensor input, uint32_t kernel_h, uint32_t kernel_w,
                                      uint32_t input_h, uint32_t input_w,
                                      uint32_t channels_per_group, uint32_t output_h,
                                      uint32_t output_w, uint32_t group, uint32_t row_len,
                                      uint32_t col_len) const;
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_CONVOLUTION_HPP_
