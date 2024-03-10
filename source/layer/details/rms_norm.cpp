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
// Created by fss on 24-2-9.
//

#include "rms_norm.hpp"
namespace kuiper_infer {
RMSNormLayer::RMSNormLayer() : ParamLayer("rms_norm") { this->weights_.resize(1); }

void RMSNormLayer::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << "This function is not implement in the RMSNorm layer！";
}

void RMSNormLayer::set_weights(const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  CHECK(weights.size() == weights_.size());
  this->weights_ = weights;
}

StatusCode RMSNormLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the rmsnorm layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the rmsnorm layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the rmsnorm "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  if (weights_.empty() || weights_.front()->empty()) {
    LOG(ERROR) << "The weight for the rmsnorm layer is missing.";
    return StatusCode::kInferParameterError;
  }

  std::shared_ptr<Tensor<float>> weight = this->weight(0);
  arma::fvec weight_vec(weight->raw_ptr(), weight->size(), false, true);
  const uint32_t batch_size = inputs.size();
#pragma omp parallel for if (batch_size > 1) num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the rmsnorm layer has an "
           "empty tensor "
        << i << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the rmsnorm "
           "layer do not match "
        << i << " th";

    const size_t size = input->size();
    arma::fvec input_vec(input->raw_ptr(), size, false, true);

    const arma::fvec& input_pow_vec = arma::pow(input_vec, 2.f);
    const float mean_value = arma::mean(input_pow_vec);
    const float norm_value = 1.f / std::sqrt(mean_value + eps_);
    arma::fvec output_vec(output->raw_ptr(), size, false, true);
    output_vec = weight_vec % (norm_value * input_vec);
  }
  return StatusCode::kSuccess;
}
}  // namespace kuiper_infer