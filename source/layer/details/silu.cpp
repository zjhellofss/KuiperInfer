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

#include "silu.hpp"
#include "activation_sse.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "tick.hpp"

namespace kuiper_infer {

SiLULayer::SiLULayer() : NonParamLayer("SiLU") {}

StatusCode SiLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the silu layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the silu layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the silu "
                  "layer do not match";
    return StatusCode::kInferInOutDimMismatch;
  }

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the silu layer has an empty tensor " << i << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the silu layer do not match " << i << " th";
    using namespace kuiper_infer::activation;
    ApplySSEActivation(ActivationType::kActivationSilu)(input, output);
  }
  return StatusCode::kSuccess;
}

StatusCode SiLULayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& silu_layer) {
  CHECK(op != nullptr) << "SiLU operator is nullptr";
  silu_layer = std::make_shared<SiLULayer>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kSiluCreateInstance("nn.SiLU", SiLULayer::CreateInstance);

}  // namespace kuiper_infer
