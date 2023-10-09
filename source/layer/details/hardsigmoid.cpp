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
#include "hardsigmoid.hpp"
#include "activation_sse.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
HardSigmoid::HardSigmoid() : NonParamLayer("HardSigmoid") {}

StatusCode HardSigmoid::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the hardsigmoid layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the hardsigmoid layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the hardsigmoid "
                  "layer do not match";
    return StatusCode::kInferInOutDimMismatch;
  }

  using namespace activation;
  ActivationFunc hardsigmoid_function = ApplySSEActivation(ActivationType::kActivationHardSigmoid);

  const uint32_t batch = inputs.size();
#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the hardsigmoid layer has an "
           "empty tensor "
        << i << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->channels(), input->rows(), input->cols());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The output and input shapes of the hardsigmoid layer do "
           "not match "
        << i << " th";
    using namespace activation;
    hardsigmoid_function(input, output);
  }
  return StatusCode::kSuccess;
}

StatusCode HardSigmoid::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                       std::shared_ptr<Layer<float>>& hardsigmoid_layer) {
  CHECK(op != nullptr) << "HardSigmoid operator is nullptr";
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kHardSigmoidCreateInstance("nn.Hardsigmoid", HardSigmoid::CreateInstance);

}  // namespace kuiper_infer
