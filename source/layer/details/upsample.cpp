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

static void CalcIndexAndLambda(int32_t input_size, int32_t output_size, float div_scale,
                               int32_t output_idx, float& lambda0, float& lambda1,
                               int32_t& input_index0, int32_t& input_index1, bool is_align_corner) {
  if (output_size == input_size) {
    input_index0 = input_index1 = output_idx;
    lambda0 = 1;
    lambda1 = 0;
  } else {
    float real_input_idx = 0.f;
    if (!is_align_corner) {
      real_input_idx = div_scale * (static_cast<float>(output_idx) + 0.5f) - 0.5f;
      if (real_input_idx < 0) {
        real_input_idx = 0;
      }
    } else {
      real_input_idx = div_scale * static_cast<float>(output_idx);
    }

    input_index0 = static_cast<int32_t>(real_input_idx);
    int32_t offset = (input_index0 < input_size - 1) ? 1 : 0;
    input_index1 = input_index0 + offset;

    lambda1 = real_input_idx - static_cast<float>(input_index0);
    lambda0 = 1.0f - lambda1;
  }
}

UpSampleLayer::UpSampleLayer(float scale_h, float scale_w, UpSampleMode mode, bool is_align_corner)
    : NonParamLayer("upsample"),
      scale_h_(scale_h),
      scale_w_(scale_w),
      mode_(mode),
      is_align_corner_(is_align_corner) {}

StatusCode UpSampleLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the upsample layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the upsample layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the upsample "
                  "layer do not match";
    return StatusCode::kInferDimMismatch;
  }
  uint32_t scale_w = static_cast<uint32_t>(scale_w_);
  uint32_t scale_h = static_cast<uint32_t>(scale_h_);

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const arma::fcube& input_data = inputs.at(i)->data();
    LOG_IF(FATAL, input_data.empty())
        << "The input tensor array in the upsample layer has an empty tensor " << i << " th";
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input_data.n_slices, input_data.n_rows * scale_h,
                                               input_data.n_cols * scale_w);
      outputs.at(i) = output;
    }
    auto& output_data = output->data();
    CHECK(output_data.n_rows == std::floor(input_data.n_rows * scale_h_))
        << "The input and output tensor height of the upsample layer do not "
           "match "
        << i << "th";
    CHECK(output_data.n_cols == std::floor(input_data.n_cols * scale_w_))
        << "The input and output tensor width of the upsample layer do not "
           "match "
        << i << "th";

    CHECK(input_data.n_slices == output_data.n_slices)
        << "The input and output tensor channel of the upsample layer do not "
           "match "
        << i << "th";

    const uint32_t channels = input_data.n_slices;
    switch (mode_) {
      case UpSampleMode::kModeNearest: {
#pragma omp parallel for
        for (uint32_t c = 0; c < channels; ++c) {
          arma::fmat& output_channel = output_data.slice(c);
          const arma::fmat& input_channel = input_data.slice(c);

          const uint32_t input_w = input_channel.n_cols;
          const uint32_t input_h = input_channel.n_rows;
          const uint32_t output_w = output_channel.n_cols;
          const uint32_t output_h = output_channel.n_rows;
          for (uint32_t w = 0; w < input_w; ++w) {
            const uint32_t dest_w = w * scale_w;
            const float* input_col_ptr = input_channel.colptr(w);
            for (uint32_t sw = 0; sw < scale_w; ++sw) {
              if (dest_w + sw >= output_w) {
                continue;
              }
              float* output_col_ptr = output_channel.colptr(dest_w + sw);
              for (uint32_t h = 0; h < input_h; ++h) {
                const uint32_t dest_h = h * scale_h;
                float* output_ptr = output_col_ptr + dest_h;
                const float input_value = *(input_col_ptr + h);
                for (uint32_t sh = 0; sh < scale_h; ++sh) {
                  if (dest_h + sh < output_h) {
                    *(output_ptr + sh) = input_value;
                  }
                }
              }
            }
          }
        }
        break;
      }
      case UpSampleMode::kModeBilinear: {
#pragma omp parallel for
        for (uint32_t c = 0; c < channels; ++c) {
          const arma::fmat& input_channel = input_data.slice(c);
          arma::fmat& output_channel = output_data.slice(c);

          const uint32_t input_w = input_channel.n_cols;
          const uint32_t input_h = input_channel.n_rows;
          const uint32_t output_w = output_channel.n_cols;
          const uint32_t output_h = output_channel.n_rows;
          float div_scale_h = 1.f;
          float div_scale_w = 1.f;
          if (!is_align_corner_) {
            div_scale_h = 1.f / scale_h_;
            div_scale_w = 1.f / scale_w_;
          } else {
            CHECK(input_h > 0 && input_w > 0);
            CHECK(output_h > 0 && output_w > 0);

            div_scale_h = static_cast<float>(input_h - 1) / static_cast<float>(output_h - 1);
            div_scale_w = static_cast<float>(input_w - 1) / static_cast<float>(output_w - 1);
          }
          for (uint32_t w = 0; w < output_w; ++w) {
            float w0_lambda = 0.f;
            float w1_lambda = 0.f;
            int32_t input_w0 = 0;
            int32_t input_w1 = 0;
            float* output_ptr = output_channel.colptr(w);
            CalcIndexAndLambda(static_cast<int32_t>(input_w), static_cast<int32_t>(output_w),
                               div_scale_w, static_cast<int32_t>(w), w0_lambda, w1_lambda, input_w0,
                               input_w1, is_align_corner_);
            const float* input_ptr0 = input_channel.colptr(input_w0);
            const float* input_ptr1 = input_channel.colptr(input_w1);
            for (uint32_t h = 0; h < output_h; ++h) {
              float h0_lambda = 0.f;
              float h1_lambda = 0.f;
              int32_t input_h0 = 0;
              int32_t input_h1 = 0;
              CalcIndexAndLambda(static_cast<int32_t>(input_h), static_cast<int32_t>(output_h),
                                 div_scale_h, static_cast<int32_t>(h), h0_lambda, h1_lambda,
                                 input_h0, input_h1, is_align_corner_);

              *(output_ptr + h) = h0_lambda * w0_lambda * (*(input_ptr0 + input_h0)) +
                                  h0_lambda * w1_lambda * (*(input_ptr1 + input_h0)) +
                                  h1_lambda * w0_lambda * (*(input_ptr0 + input_h1)) +
                                  h1_lambda * w1_lambda * (*(input_ptr1 + input_h1));
            }
          }
        }
        break;
      }
      default: {
        LOG(FATAL) << "Unsupported upsample mode in the upsample layer: " << int32_t(mode_);
        break;
      }
    }
  }
  return StatusCode::kSuccess;
}

StatusCode UpSampleLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                         std::shared_ptr<Layer<float>>& upsample_layer) {
  if (!op) {
    LOG(ERROR) << "The upsample operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the upsample layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (params.find("scale_factor") == params.end()) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return StatusCode::kParseParameterError;
  }

  auto scales = std::dynamic_pointer_cast<RuntimeParameterFloatArray>(params.at("scale_factor"));
  if (scales == nullptr) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return StatusCode::kParseParameterError;
  }

  if (scales->value.size() != 2) {
    LOG(ERROR) << "Scale factor need two dimension";
    return StatusCode::kParseParameterError;
  }

  if (params.find("mode") == params.end()) {
    LOG(ERROR) << "Can not find the mode parameter";
    return StatusCode::kParseParameterError;
  }

  UpSampleMode mode;
  auto mode_param = std::dynamic_pointer_cast<RuntimeParameterString>(params.at("mode"));
  if (mode_param->value == "nearest") {
    mode = UpSampleMode::kModeNearest;
  } else if (mode_param->value == "bilinear") {
    mode = UpSampleMode::kModeBilinear;
  } else {
    LOG(ERROR) << "The mode " << mode_param->value << " is not supported!";
    return StatusCode::kParseParameterError;
  }

  bool is_align_corner = false;
  if (params.find("align_corners") != params.end()) {
    auto align_corner_param =
        std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("align_corners"));
    if (!align_corner_param) {
      return StatusCode::kParseParameterError;
    }
    is_align_corner = align_corner_param->value;
  }

  const float scale_h = scales->value.at(0);
  const float scale_w = scales->value.at(1);
  // scale放大的倍数大于0
  if (scale_h <= 0 || scale_w <= 0) {
    LOG(ERROR) << "The parameter scale height and scale width should be greater than zero.";
    return StatusCode::kParseParameterError;
  }

  upsample_layer = std::make_shared<UpSampleLayer>(scale_h, scale_w, mode, is_align_corner);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kUpSamplerCreateInstance(UpSampleLayer::CreateInstance, "nn.Upsample",
                                                "F.upsample");
}  // namespace kuiper_infer
