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

#ifndef KUIPER_INFER_SOURCE_LAYER_LINEAR_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LINEAR_HPP_
#include "layer/abstract/layer.hpp"
#include "layer/abstract/param_layer.hpp"

namespace kuiper_infer {
class LinearLayer : public ParamLayer {
 public:
  //  explicit LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim, uint32_t out_dim,
  //  bool use_bias = true);
  explicit LinearLayer(int32_t in_features, int32_t out_features, bool use_bias);

  StatusCode Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static StatusCode CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                   std::shared_ptr<Layer<float>>& linear_layer);

 private:
  int32_t in_features_ = 0;
  int32_t out_features_ = 0;
  bool use_bias_ = false;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_LINEAR_HPP_
