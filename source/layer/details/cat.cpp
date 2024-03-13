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
#include "cat.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
CatLayer::CatLayer(int32_t dim) : NonParamLayer("cat"), dim_(dim) {}

StatusCode CatLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                             std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the cat layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension parameter of cat layer is error";
    return StatusCode::kInferParameterError;
  }

  const uint32_t output_size = outputs.size();
  if (inputs.size() % output_size != 0) {
    LOG(ERROR) << "The input and output tensor array size of cat layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  const uint32_t packet_size = inputs.size() / output_size;
#pragma omp parallel for num_threads(outputs.size())
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    uint32_t copy_channel_offset = 0;
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    for (uint32_t j = i; j < inputs.size(); j += output_size) {
      const std::shared_ptr<Tensor<float>>& input = inputs.at(j);
      CHECK(input != nullptr && !input->empty()) << "The input tensor array in the cat layer has "
                                                    "an empty tensor "
                                                 << j << " th";
      const uint32_t in_rows = input->rows();
      const uint32_t in_cols = input->cols();
      const uint32_t in_channels = input->channels();
      CHECK(in_rows == input->rows() && in_cols == input->cols())
          << "The input tensor array in the cat layer "
             "has an incorrectly sized tensor "
          << j << " th";

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(in_channels * packet_size, in_rows, in_cols);
        outputs.at(i) = output;
      }
      CHECK(output->channels() == in_channels * packet_size && output->rows() == in_rows &&
            output->cols() == in_cols)
          << "The output tensor array in the cat layer "
             "has an incorrectly sized tensor "
          << i << " th";
      const uint32_t plane_size = in_rows * in_cols;
      memcpy(output->raw_ptr(copy_channel_offset * plane_size), input->raw_ptr(),
             sizeof(float) * plane_size * in_channels);
      copy_channel_offset += input->channels();
    }
  }
  return StatusCode::kSuccess;
}

StatusCode CatLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                    std::shared_ptr<Layer<float>>& cat_layer) {
  if (!op) {
    LOG(ERROR) << "The cat operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the cat layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParameterError;
  }

  auto dim_param = std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return StatusCode::kParseParameterError;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kCatCreateInstance(CatLayer::CreateInstance, "torch.cat");
}  // namespace kuiper_infer