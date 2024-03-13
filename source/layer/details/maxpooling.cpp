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

// Created by fss on 22-11-18.

#include "maxpooling.hpp"
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
namespace kuiper_infer {

MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w, uint32_t pooling_size_h,
                                 uint32_t pooling_size_w, uint32_t stride_h, uint32_t stride_w)
    : NonParamLayer("MaxPooling"),
      padding_h_(padding_h),
      padding_w_(padding_w),
      pooling_size_h_(pooling_size_h),
      pooling_size_w_(pooling_size_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {
  CHECK_GT(stride_h_, 0);
  CHECK_GT(stride_w_, 0);
  CHECK_GT(pooling_size_h_, 0);
  CHECK_GT(pooling_size_w_, 0);
}

StatusCode MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the maxpooling layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the maxpooling layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the maxpooling "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }

  if (!pooling_size_h_ || !pooling_size_w_) {
    LOG(ERROR) << "The pooling size in the maxpooling layer should be greater "
                  "than zero";
    return StatusCode::kInferParameterError;
  }

  if (!stride_w_ || !stride_h_) {
    LOG(ERROR) << "The stride in the maxpooling layer should be greater "
                  "than zero";
    return StatusCode::kInferParameterError;
  }

  const uint32_t batch = inputs.size();
  const uint32_t pooling_h = pooling_size_h_;
  const uint32_t pooling_w = pooling_size_w_;

#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    CHECK(input_data != nullptr && !input_data->empty())
        << "The input tensor array in the max pooling layer has an "
           "empty tensor "
        << i << "th";

    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_padded_h = input_data->rows() + 2 * padding_h_;
    const uint32_t input_padded_w = input_data->cols() + 2 * padding_w_;

    const uint32_t input_c = input_data->channels();

    const uint32_t output_h =
        uint32_t(std::floor((int32_t(input_padded_h) - int32_t(pooling_h)) / stride_h_ + 1));
    const uint32_t output_w =
        uint32_t(std::floor((int32_t(input_padded_w) - int32_t(pooling_w)) / stride_w_ + 1));

    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    if (output_data == nullptr || output_data->empty()) {
      output_data = std::make_shared<Tensor<float>>(input_c, output_h, output_w);
      outputs.at(i) = output_data;
    }

    CHECK(output_data->rows() == output_h && output_data->cols() == output_w &&
          output_data->channels() == input_c)
        << "The output tensor array in the max pooling layer "
           "has an incorrectly sized tensor "
        << i << "th";

    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_channel = input_data->slice(ic);
      arma::fmat& output_channel = output_data->slice(ic);
      for (uint32_t c = 0; c < input_padded_w - pooling_w + 1; c += stride_w_) {
        uint32_t output_col = uint32_t(c / stride_w_);
        for (uint32_t r = 0; r < input_padded_h - pooling_h + 1; r += stride_h_) {
          uint32_t output_row = uint32_t(r / stride_h_);
          float* output_channel_ptr = output_channel.colptr(output_col);
          float max_value = std::numeric_limits<float>::lowest();
          for (uint32_t w = 0; w < pooling_w; ++w) {
            const float* col_ptr = input_channel.colptr(c + w - padding_w_) + r;
            for (uint32_t h = 0; h < pooling_h; ++h) {
              float current_value = 0.f;
              if ((h + r >= padding_h_ && w + c >= padding_w_) &&
                  (h + r < input_h + padding_h_ && w + c < input_w + padding_w_)) {
                current_value = *(col_ptr + h - padding_h_);
              } else {
                current_value = std::numeric_limits<float>::lowest();
              }
              max_value = std::max(max_value, current_value);
            }
          }
          *(output_channel_ptr + output_row) = max_value;
        }
      }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode MaxPoolingLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                           std::shared_ptr<Layer<float>>& max_layer) {
  if (!op) {
    LOG(ERROR) << "The maxpooling operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the maxpooling layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return StatusCode::kParseParameterError;
  }

  auto stride = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return StatusCode::kParseParameterError;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return StatusCode::kParseParameterError;
  }

  auto padding = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return StatusCode::kParseParameterError;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return StatusCode::kParseParameterError;
  }

  auto kernel_size = std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("kernel_size"));
  if (!kernel_size) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return StatusCode::kParseParameterError;
  }
  const auto& padding_values = padding->value;
  const auto& stride_values = stride->value;
  const auto& kernel_values = kernel_size->value;

  const uint32_t dims = 2;
  if (padding_values.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return StatusCode::kParseParameterError;
  }

  if (stride_values.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return StatusCode::kParseParameterError;
  }

  if (kernel_values.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return StatusCode::kParseParameterError;
  }

  max_layer = std::make_shared<MaxPoolingLayer>(padding_values.at(0), padding_values.at(1),
                                                kernel_values.at(0), kernel_values.at(1),
                                                stride_values.at(0), stride_values.at(1));

  return StatusCode::kSuccess;
}

LayerRegistererWrapper kMaxPoolingCreateInstance(MaxPoolingLayer::CreateInstance, "nn.MaxPool2d");

}  // namespace kuiper_infer