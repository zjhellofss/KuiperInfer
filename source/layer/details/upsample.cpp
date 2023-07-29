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
#include "upsample.hpp"
#include <cmath>
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {

UpSampleLayer::UpSampleLayer(float scale_h, float scale_w, UpSampleMode mode)
    : NonParamLayer("upsample"),
      scale_h_(scale_h),
      scale_w_(scale_w),
      mode_(mode) {}

InferStatus UpSampleLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the upsample layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output tensor array size of the upsample layer do "
           "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  auto test_scale_factor = [](uint32_t origin, float scale_factor) {
    float result = origin * scale_factor;
    if (std::abs(result - std::round(result)) > 1e-4f) {
      LOG(ERROR) << "The input scale_factor is wrong";
    }
  };

  LOG_IF(FATAL, this->mode_ != UpSampleMode::kModeNearest)
      << "Unsupported upsample mode: " << int(mode_);

  for (uint32_t i = 0; i < inputs.size(); ++i) {
    const sftensor& input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR)
          << "The input tensor array in the upsample layer has an empty tensor "
          << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    test_scale_factor(inputs.at(i)->data().n_rows, scale_h_);
    test_scale_factor(inputs.at(i)->data().n_cols, scale_w_);
  }

  const uint32_t batch_size = inputs.size();
  const uint32_t scale_w = uint32_t(scale_w_);
  const uint32_t scale_h = uint32_t(scale_h_);

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const arma::fcube& input_data = inputs.at(i)->data();
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(
          input_data.n_slices, uint32_t(input_data.n_rows * scale_h),
          uint32_t(input_data.n_cols * scale_w));
      outputs.at(i) = output;
    }
    auto& output_data = output->data();
    CHECK(output_data.n_rows == input_data.n_rows * scale_h)
        << "The input and output tensor height of the upsample layer do not "
           "match "
        << i << "th";
    CHECK(output_data.n_cols == input_data.n_cols * scale_w)
        << "The input and output tensor width of the upsample layer do not "
           "match "
        << i << "th";

    CHECK(input_data.n_slices == output_data.n_slices)
        << "The input and output tensor channel of the upsample layer do not "
           "match "
        << i << "th";

    const uint32_t channels = input_data.n_slices;
    for (uint32_t c = 0; c < channels; ++c) {
      const arma::fmat& input_channel = input_data.slice(c);
      arma::fmat& output_channel = output_data.slice(c);

      const uint32_t input_w = input_channel.n_cols;
      const uint32_t input_h = input_channel.n_rows;
      const uint32_t output_w = output_channel.n_cols;
      const uint32_t output_h = output_channel.n_rows;

      for (uint32_t w = 0; w < input_w; ++w) {
        const float* input_col_ptr = input_channel.colptr(w);
        const uint32_t scaled_w = w * scale_w;
        for (uint32_t sw = 0; sw < scale_w; ++sw) {
          if (scaled_w + sw >= output_w) {
            continue;
          }
          float* output_col_ptr = output_channel.colptr(scaled_w + sw);
          for (uint32_t h = 0; h < input_h; ++h) {
            const uint32_t scaled_h = h * scale_h;
            float* output_ptr = output_col_ptr + scaled_h;
            float input_value = *(input_col_ptr + h);
            for (uint32_t sh = 0; sh < scale_h; ++sh) {
              if (scaled_h + sh < output_h) {
                *(output_ptr + sh) = input_value;
              }
            }
          }
        }
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus UpSampleLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& upsample_layer) {
  CHECK(op != nullptr) << "Upsample operator is null";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";
  if (params.find("scale_factor") == params.end()) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return ParseParameterAttrStatus::kParameterMissingScale;
  }

  const auto& scale_param = params.at("scale_factor");
  auto scales = std::dynamic_pointer_cast<RuntimeParameterFloatArray>(
      params.at("scale_factor"));
  if (scales == nullptr) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return ParseParameterAttrStatus::kParameterMissingScale;
  }
  CHECK(scales->value.size() == 2) << "Scale factor need two dimension";

  if (params.find("mode") == params.end()) {
    LOG(ERROR) << "Can not find the mode parameter";
    return ParseParameterAttrStatus::kParameterMissingResizeMode;
  }

  auto mode =
      std::dynamic_pointer_cast<RuntimeParameterString>(params.at("mode"));
  CHECK(mode->value == "nearest")
      << "The mode " << mode->value << " is not supported!";

  float scale_h = scales->value.at(0);
  float scale_w = scales->value.at(1);
  upsample_layer = std::make_shared<UpSampleLayer>(scale_h, scale_w);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kUpSamplerGetInstance("nn.Upsample",
                                             UpSampleLayer::GetInstance);
}  // namespace kuiper_infer
