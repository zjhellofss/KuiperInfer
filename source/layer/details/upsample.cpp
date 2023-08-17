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

static void CalcIndexAndLambda(int32_t input_size, int32_t output_size,
                               float div_scale, int32_t output_idx,
                               float& lambda0, float& lambda1,
                               int32_t& input_index0, int32_t& input_index1) {
  if (output_size == input_size) {
    input_index0 = input_index1 = output_idx;
    lambda0 = 1;
    lambda1 = 0;
  } else {
    float real_input_idx =
        div_scale * (static_cast<float>(output_idx) + 0.5f) - 0.5f;
    if (real_input_idx < 0) {
      real_input_idx = 0;
    }

    input_index0 = static_cast<int>(real_input_idx);
    int32_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;

    lambda1 = real_input_idx - static_cast<float>(input_index0);
    lambda0 = 1.0f - lambda1;
  }
}

UpSampleLayer::UpSampleLayer(uint32_t scale_h, uint32_t scale_w,
                             UpSampleMode mode)
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

  auto test_scale_factor = [](uint32_t origin, uint32_t scale_factor) {
    float result = static_cast<float>(origin * scale_factor);
    if (std::abs(result - std::round(result)) > 1e-5f) {
      LOG(ERROR) << "The input scale_factor is wrong";
    }
  };

  LOG_IF(FATAL, this->mode_ != UpSampleMode::kModeNearest &&
                    this->mode_ != UpSampleMode::kModeBilinear)
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

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const arma::fcube& input_data = inputs.at(i)->data();
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input_data.n_slices,
                                               input_data.n_rows * scale_h_,
                                               input_data.n_cols * scale_w_);
      outputs.at(i) = output;
    }
    auto& output_data = output->data();
    CHECK(output_data.n_rows == input_data.n_rows * scale_h_)
        << "The input and output tensor height of the upsample layer do not "
           "match "
        << i << "th";
    CHECK(output_data.n_cols == input_data.n_cols * scale_w_)
        << "The input and output tensor width of the upsample layer do not "
           "match "
        << i << "th";

    CHECK(input_data.n_slices == output_data.n_slices)
        << "The input and output tensor channel of the upsample layer do not "
           "match "
        << i << "th";

    const float div_scale_h = 1.f / static_cast<float>(scale_h_);
    const float div_scale_w = 1.f / static_cast<float>(scale_w_);
    const uint32_t channels = input_data.n_slices;
    if (mode_ == UpSampleMode::kModeNearest) {
#pragma omp parallel for
      for (uint32_t c = 0; c < channels; ++c) {
        const arma::fmat& input_channel = input_data.slice(c);
        arma::fmat& output_channel = output_data.slice(c);

        const uint32_t input_w = input_channel.n_cols;
        const uint32_t input_h = input_channel.n_rows;
        const uint32_t output_w = output_channel.n_cols;
        const uint32_t output_h = output_channel.n_rows;
        for (uint32_t w = 0; w < input_w; ++w) {
          const float* input_col_ptr = input_channel.colptr(w);
          const uint32_t scaled_w = w * scale_w_;
          for (uint32_t sw = 0; sw < scale_w_; ++sw) {
            if (scaled_w + sw >= output_w) {
              continue;
            }
            float* output_col_ptr = output_channel.colptr(scaled_w + sw);
            for (uint32_t h = 0; h < input_h; ++h) {
              const uint32_t scaled_h = h * scale_h_;
              float* output_ptr = output_col_ptr + scaled_h;
              float input_value = *(input_col_ptr + h);
              for (uint32_t sh = 0; sh < scale_h_; ++sh) {
                if (scaled_h + sh < output_h) {
                  *(output_ptr + sh) = input_value;
                }
              }
            }
          }
        }
      }
    } else {
#pragma omp parallel for
      for (uint32_t c = 0; c < channels; ++c) {
        const arma::fmat& input_channel = input_data.slice(c);
        arma::fmat& output_channel = output_data.slice(c);

        const uint32_t input_w = input_channel.n_cols;
        const uint32_t input_h = input_channel.n_rows;
        const uint32_t output_w = output_channel.n_cols;
        const uint32_t output_h = output_channel.n_rows;
        for (uint32_t w = 0; w < output_w; ++w) {
          float* output_ptr = output_channel.colptr(w);
          float w0_lambda = 0.f;
          float w1_lambda = 0.f;
          int32_t input_w0 = 0;
          int32_t input_w1 = 0;
          CalcIndexAndLambda(static_cast<int32_t>(input_w),
                             static_cast<int32_t>(output_w), div_scale_w,
                             static_cast<int32_t>(w), w0_lambda, w1_lambda,
                             input_w0, input_w1);
          const float* input_ptr0 = input_channel.colptr(input_w0);
          const float* input_ptr1 = input_channel.colptr(input_w1);
          for (uint32_t h = 0; h < output_h; ++h) {
            float h0_lambda = 0.f;
            float h1_lambda = 0.f;
            int32_t input_h0 = 0;
            int32_t input_h1 = 0;
            CalcIndexAndLambda(static_cast<int32_t>(input_h),
                               static_cast<int32_t>(output_h), div_scale_h,
                               static_cast<int32_t>(h), h0_lambda, h1_lambda,
                               input_h0, input_h1);

            *(output_ptr + h) =
                h0_lambda * w0_lambda * (*(input_ptr0 + input_h0)) +
                h0_lambda * w1_lambda * (*(input_ptr1 + input_h0)) +
                h1_lambda * w0_lambda * (*(input_ptr0 + input_h1)) +
                h1_lambda * w1_lambda * (*(input_ptr1 + input_h1));
          }
        }
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus UpSampleLayer::CreateInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& upsample_layer) {
  CHECK(op != nullptr) << "Upsample operator is null";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";
  if (params.find("scale_factor") == params.end()) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return ParseParameterAttrStatus::kParameterMissingScale;
  }

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

  auto mode_param =
      std::dynamic_pointer_cast<RuntimeParameterString>(params.at("mode"));

  UpSampleMode mode;
  if (mode_param->value == "nearest") {
    mode = UpSampleMode::kModeNearest;
  } else if (mode_param->value == "bilinear") {
    mode = UpSampleMode::kModeBilinear;
  } else {
    LOG(FATAL) << "The mode " << mode_param->value << " is not supported!";
  }

  if (params.find("align_corners") != params.end()) {
    auto align_corner_param = std::dynamic_pointer_cast<RuntimeParameterBool>(
        params.at("align_corners"));
    if (!align_corner_param) {
      return ParseParameterAttrStatus::kParameterMissingAlignCorner;
    }
    bool align_corner = align_corner_param->value;
    CHECK_EQ(align_corner, false);
  }

  float scale_h = scales->value.at(0);
  float scale_w = scales->value.at(1);
  // scale放大的倍数大于0
  CHECK_GT(scale_h, 0.f);
  CHECK_GT(scale_w, 0.f);

  // scale放大的倍数必须是整数
  CHECK_LE(scale_h - static_cast<uint32_t>(scale_h), 1e-5f);
  CHECK_LE(scale_w - static_cast<uint32_t>(scale_w), 1e-5f);

  upsample_layer = std::make_shared<UpSampleLayer>(
      static_cast<uint32_t>(scale_h), static_cast<uint32_t>(scale_w), mode);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kUpSamplerCreateInstance("nn.Upsample",
                                                UpSampleLayer::CreateInstance);
LayerRegistererWrapper kUpSamplerFCreateInstance("F.upsample",
                                                 UpSampleLayer::CreateInstance);
}  // namespace kuiper_infer
