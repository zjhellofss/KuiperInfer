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

// Created by fss on 22-11-12.

#ifndef KUIPER_INFER_SOURCE_LAYER_MAX_POOLING_
#define KUIPER_INFER_SOURCE_LAYER_MAX_POOLING_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class MaxPoolingLayer : public NonParamLayer {
 public:
  explicit MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                           uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& max_layer);

 private:
  uint32_t padding_h_ = 0;
  uint32_t padding_w_ = 0;
  uint32_t pooling_size_h_ = 0;
  uint32_t pooling_size_w_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_MAX_POOLING_
