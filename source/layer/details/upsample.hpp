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

// Created by fss on 22-12-25.

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
#include "layer/abstract/non_param_layer.hpp"

namespace kuiper_infer {
enum class UpSampleMode {
  kModeNearest = 0,
  kModeBilinear = 1,  // 目前上采样支持这两种
};

class UpSampleLayer : public NonParamLayer {
 public:
  explicit UpSampleLayer(uint32_t scale_h, uint32_t scale_w,
                         UpSampleMode mode = UpSampleMode::kModeNearest);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus CreateInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& upsample_layer);

 private:
  uint32_t scale_h_ = 1;
  uint32_t scale_w_ = 1;
  UpSampleMode mode_ = UpSampleMode::kModeNearest;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_UPSAMPLE_HPP_
