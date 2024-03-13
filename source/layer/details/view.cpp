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
#include "view.hpp"
#include <glog/logging.h>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

StatusCode ViewLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                              std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the view layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the view layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the view "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  const uint32_t batch_size = inputs.size();
  if (shapes_.empty() || (shapes_.front() != -1 && shapes_.front() != batch_size)) {
    LOG(ERROR) << "The shape parameter in the view layer has an incorrectly size! ";
    return StatusCode::kInferParameterError;
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    CHECK(input_data != nullptr && !input_data->empty())
        << "The input tensor array in the view layer has an empty tensor " << i << " th";

    // 检查形状中-1的数量，最多只可以存在一个
    size_t current_size = 1;
    int32_t dynamic_index = -1;
    std::vector<uint32_t> shapes;
    const size_t total_size = input_data->size();
    for (uint32_t j = 1; j < shapes_.size(); ++j) {
      CHECK(shapes_.at(j) == -1 || shapes_.at(j) > 0);
      if (shapes_.at(j) == -1) {
        CHECK(dynamic_index == -1)
            << "Having two minus value in shape parameters of the view layer";
        dynamic_index = static_cast<int32_t>(j);
      } else {
        current_size *= shapes_.at(j);
        shapes.push_back(shapes_.at(j));
      }
    }

    CHECK(dynamic_index == -1 || dynamic_index == shapes_.size() - 1)
        << "-1 appears in the wrong dimension, it can only be on the last "
           "dimension";
    if (dynamic_index != -1) {
      CHECK(total_size >= current_size);
      shapes.push_back(uint32_t(total_size / current_size));
    }

    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    output_data = TensorClone(input_data);
    CHECK(input_data->size() == output_data->size());
    outputs.at(i) = output_data;

    output_data->Reshape(shapes, true);
  }
  return StatusCode::kSuccess;
}

StatusCode ViewLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                     std::shared_ptr<Layer<float>>& view_layer) {
  if (!op) {
    LOG(ERROR) << "The view operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the view layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (params.find("shape") == params.end()) {
    LOG(ERROR) << "View layer missing shape";
    return StatusCode::kParseParameterError;
  }

  auto shape = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("shape"));
  if (!shape) {
    LOG(ERROR) << "View layer missing shape";
    return StatusCode::kParseParameterError;
  }
  view_layer = std::make_shared<ViewLayer>(shape->value);
  return StatusCode::kSuccess;
}

ViewLayer::ViewLayer(std::vector<int32_t> shapes)
    : NonParamLayer("view"), shapes_(std::move(shapes)) {}

LayerRegistererWrapper kViewCreateInstance(ViewLayer::CreateInstance, "Tensor.view");

}  // namespace kuiper_infer
