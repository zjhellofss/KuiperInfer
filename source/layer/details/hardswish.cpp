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
    
// Created by fss on 22-12-12.
#include "hardswish.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
HardSwishLayer::HardSwishLayer() : NonParamLayer("HardSwish") {}

InferStatus HardSwishLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the hardswish layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the hardswish "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch = inputs.size();
#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the hardswish layer has an empty tensor "
        << i << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The output tensor array in the hardswish layer has an "
                     "empty tensor "
                  << i << " th";
      output = std::make_shared<Tensor<float>>(input->channels(), input->rows(),
                                               input->cols());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the hardswish layer do not "
           "match "
        << i << " th";

    for (uint32_t j = 0; j < input->size(); ++j) {
      float val = input->index(j);
      float result = 0.f;
      if (val <= -3.f) {
        result = 0.f;
      } else if (val >= 3.f) {
        result = val;
      } else {
        result = val * (val + 3) / 6;
      }
      output->index(j) = result;
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus HardSwishLayer::CreateInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& hardswish_layer) {
  CHECK(op != nullptr) << "HardSwishLayer operator is nullptr";
  hardswish_layer = std::make_shared<HardSwishLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kHardSwishCreateInstance("nn.Hardswish",
                                             HardSwishLayer::CreateInstance);

}  // namespace kuiper_infer