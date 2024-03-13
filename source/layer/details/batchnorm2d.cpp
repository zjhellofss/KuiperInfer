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

// Created by fss on 22-11-17.
#include "batchnorm2d.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

StatusCode BatchNorm2dLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the batchnorm2d layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the batchnorm2d layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the batchnorm2d "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  const uint32_t mean_value_size = this->weights_.size();
  const uint32_t bias_value_size = this->bias_.size();
  if (mean_value_size != bias_value_size) {
    LOG(ERROR) << "The batchnorm2d layer do not have the same number of mean "
                  "values and bias values";
    return StatusCode::kInferParameterError;
  }

  if (this->affine_weight_.size() != this->weights().size()) {
    LOG(ERROR) << "The batchnorm2d layer do not have the same number of mean "
                  "values and affine weight";
    return StatusCode::kInferParameterError;
  }

  if (this->affine_bias_.size() != this->affine_weight_.size()) {
    LOG(ERROR) << "The batchnorm2d layer do not have the same number of affine "
                  "weight and affine bias";
    return StatusCode::kInferParameterError;
  }
  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the batchnorm2d layer has an "
           "empty tensor "
        << b << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(b) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the batchnorm2d "
           "layer do not match "
        << b << " th";

    for (uint32_t i = 0; i < mean_value_size; ++i) {
      CHECK(weights_.at(i)->size() == 1 && bias_.at(i)->size() == 1);
      const float mean_value = weights_.at(i)->index(0);
      CHECK(input->channels() > i)
          << "In the batchnorm2d layer, too few channels for input tensor " << b << " th";

      const float var_value = std::sqrt(bias_.at(i)->index(0) + eps_);
      output->slice(i) =
          ((input->slice(i) - mean_value) / var_value) * affine_weight_.at(i) + affine_bias_.at(i);
    }
  }
  return StatusCode::kSuccess;
}

StatusCode BatchNorm2dLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                            std::shared_ptr<Layer<float>>& batch_layer) {
  if (!op) {
    LOG(ERROR) << "The batchnorm operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the batchnorm layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (params.find("eps") == params.end()) {
    LOG(ERROR) << "Can not find the eps parameter";
    return StatusCode::kParseParameterError;
  }

  auto eps = std::dynamic_pointer_cast<RuntimeParameterFloat>(params.at("eps"));
  if (!eps) {
    LOG(ERROR) << "Can not find the eps parameter";
    return StatusCode::kParseParameterError;
  }

  if (params.find("num_features") == params.end()) {
    LOG(ERROR) << "Can not find the num features parameter";
    return StatusCode::kParseParameterError;
  }

  auto num_features = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("num_features"));
  if (!num_features) {
    LOG(ERROR) << "Can not find the num features parameter";
    return StatusCode::kParseParameterError;
  }

  // load weights
  const auto& attrs = op->attribute;
  CHECK(!attrs.empty()) << "Operator attributes is empty";

  if (attrs.find("running_mean") == attrs.end()) {
    LOG(ERROR) << "Can not find the running mean attribute";
    return StatusCode::kParseWeightError;
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the affine weight attribute";
    return StatusCode::kParseWeightError;
  }
  if (attrs.find("bias") == attrs.end()) {
    LOG(ERROR) << "Can not find the affine bias attribute";
    return StatusCode::kParseWeightError;
  }

  const std::vector<float>& affine_weight = attrs.at("weight")->get<float>();
  const std::vector<float>& affine_bias = attrs.at("bias")->get<float>();
  batch_layer = std::make_shared<BatchNorm2dLayer>(num_features->value, eps->value, affine_weight,
                                                   affine_bias);

  const auto& mean_attr = attrs.at("running_mean");
  const std::vector<float>& mean = mean_attr->get<float>();
  if (mean.empty()) {
    LOG(ERROR) << "The running mean weight in the batchnorm layer is empty!";
    return StatusCode::kParseWeightError;
  }
  batch_layer->set_weights(mean);

  if (attrs.find("running_var") == attrs.end()) {
    LOG(ERROR) << "Can not find the running var weight.";
    return StatusCode::kParseWeightError;
  }

  const auto& var_attr = attrs.at("running_var");
  const std::vector<float>& var = var_attr->get<float>();
  if (var.empty()) {
    LOG(ERROR) << "The running var weight in the batchnorm layer is empty!";
    return StatusCode::kParseWeightError;
  }
  batch_layer->set_bias(var);
  return StatusCode::kSuccess;
}

BatchNorm2dLayer::BatchNorm2dLayer(uint32_t num_features, float eps,
                                   std::vector<float> affine_weight, std::vector<float> affine_bias)
    : ParamLayer("Batchnorm"),
      affine_weight_(std::move(affine_weight)),
      affine_bias_(std::move(affine_bias)),
      eps_(eps) {
  this->InitWeightParam(num_features, 1, 1, 1);
  this->InitBiasParam(num_features, 1, 1, 1);
}

LayerRegistererWrapper kBatchNorm2dCreateInstance(BatchNorm2dLayer::CreateInstance,
                                                  "nn.BatchNorm2d");

}  // namespace kuiper_infer