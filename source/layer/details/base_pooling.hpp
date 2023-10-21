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
// Created by fss on 23-10-20.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_BASE_POOLING_HPP
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_BASE_POOLING_HPP
#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {

enum class PoolingType {
  kMaxPooling = 1,
  kAveragePooling = 2,
};

class BasePoolingLayer : public NonParamLayer {
 protected:
  explicit BasePoolingLayer(std::string layer_name) : NonParamLayer(std::move(layer_name)) {}

  void ComputeOutput(const sftensor input_data, sftensor output_data, uint32_t input_padded_h,
                     uint32_t input_padded_w, uint32_t input_c, uint32_t pooling_h,
                     uint32_t pooling_w, uint32_t padding_h_, uint32_t padding_w_,
                     uint32_t stride_h_, uint32_t stride_w_,
                     kuiper_infer::PoolingType pooling_typ) const;

 private:
  float ReduceMax(const std::vector<float>& window_values) const;

  float ReduceAverage(const std::vector<float>& window_values) const;
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_BASE_POOLING_HPP
