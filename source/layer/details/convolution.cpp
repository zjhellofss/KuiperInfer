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

// Created by fss on 22-11-13.

#include "convolution.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"
#include "utils/math/fmath.hpp"

namespace kuiper_infer {
void DeconvolutionLayer::set_weights(
    const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL)
      << "The set weights function does not support this convolution type: "
      << int32_t(conv_type_);
}

void DeconvolutionLayer::set_weights(const std::vector<float>& weights) {
  const uint32_t kernel_count = this->weights_.size();
  CHECK(kernel_count > 0);
  const uint32_t kernel_count_group = kernel_count / groups_;
  const uint32_t kernel_channel = this->weights_.at(0)->channels();
  uint32_t kernel_height = this->weights_.at(0)->rows();
  uint32_t kernel_width = this->weights_.at(0)->cols();
  if (dilation_h_ > 1) {
    kernel_height = (kernel_height + dilation_h_ - 1) / dilation_h_;
  }

  if (dilation_w_ > 1) {
    kernel_width = (kernel_width + dilation_w_ - 1) / dilation_w_;
  }

  const uint32_t kernel_hw = kernel_height * kernel_width;
  const uint32_t kernel_nhw = kernel_count_group * kernel_hw;
  const uint32_t kernel_plane = kernel_channel * kernel_nhw;

  for (uint32_t group = 0; group < groups_; ++group) {
    // sub_weights表示一个group内所有卷积核的权重值
    std::vector<float> sub_weights(kernel_plane);
    std::copy(weights.data() + group * kernel_plane,
              weights.data() + (group + 1) * kernel_plane, sub_weights.begin());
    for (uint32_t kg = 0; kg < kernel_count_group; ++kg) {
      const uint32_t channel_offset = kg * kernel_hw;
      const uint32_t kernel_idx = group * kernel_count_group + kg;
      /*
       * 卷积核权重摆放的顺序是c n h w， 需要将它调整到n c h w
       * 其中n表示卷积核次序，kernel_idx = group * kernel_count_group + kg;
       * origin_pixel_idx = ic * kernel_nhw (nhw) + kg(n) * kernel_hw + ...
       */
      for (uint32_t ic = 0; ic < kernel_channel; ++ic) {
        const uint32_t kernel_offset = ic * kernel_nhw;
        arma::fmat& kernel_channel_mat =
            this->weights_.at(kernel_idx)->slice(ic);

        for (uint32_t kw = 0; kw < kernel_width; ++kw) {
          uint32_t kw_dilation = kw * dilation_w_;
          float* kernel_ptr = kernel_channel_mat.colptr(kw_dilation);
          for (uint32_t kh = 0; kh < kernel_height; ++kh) {
            uint32_t kh_dilation = kh * dilation_h_;
            *(kernel_ptr + kh_dilation) = sub_weights.at(
                kernel_offset + channel_offset + kh * kernel_width + kw);
          }
        }
      }
    }
  }
}

void DeconvolutionLayer::InitIm2ColWeight() {}

void DeconvolutionLayer::ComputeOutput(sftensor input, sftensor output_tensor,
                                       uint32_t kernel_h, uint32_t kernel_w,
                                       uint32_t kernel_count_group,
                                       uint32_t input_h, uint32_t input_w,
                                       uint32_t channels_per_group,
                                       uint32_t output_h, uint32_t output_w,
                                       uint32_t group) const {
#pragma omp parallel for
  for (uint32_t k = 0; k < kernel_count_group; ++k) {
    const arma::fmat& gemm_result =
        DeconvGemm(input, input_h, input_w, channels_per_group, group, k,
                   kernel_count_group);
    DeconvCol2ImBias(gemm_result, output_tensor, input_h, input_w, group, k,
                     kernel_count_group, kernel_h, kernel_w, output_h,
                     output_w);
  }
}

std::pair<uint32_t, uint32_t> DeconvolutionLayer::ComputeOutputSize(
    const uint32_t input_h, const uint32_t input_w, const uint32_t kernel_h,
    const uint32_t kernel_w) const {
  uint32_t output_h = 0;
  uint32_t output_w = 0;

  output_h = (input_h - 1) * stride_h_ + kernel_h + output_padding_h_;
  output_w = (input_w - 1) * stride_w_ + kernel_w + output_padding_w_;
  CHECK(output_h > 2 * padding_h_ && output_w > 2 * padding_w_);
  output_h -= 2 * padding_h_;
  output_w -= 2 * padding_w_;
  return {output_h, output_w};
}

arma::fmat DeconvolutionLayer::DeconvGemm(sftensor input, uint32_t input_h,
                                          uint32_t input_w,
                                          uint32_t channels_per_group,
                                          uint32_t group, uint32_t kernel_index,
                                          uint32_t kernel_count_group) const {
  CHECK(input != nullptr && !input->empty());

  kernel_index = kernel_index + group * kernel_count_group;
  CHECK(kernel_index < this->weights_.size());

  sftensor group_kernel = this->weights_.at(kernel_index);
  CHECK(group_kernel != nullptr && !group_kernel->empty());

  uint32_t input_hw = input_h * input_w;
  uint32_t kernel_hw = group_kernel->rows() * group_kernel->cols();

  arma::fmat multi_input_channel(
      input->matrix_raw_ptr(group * channels_per_group), input_hw,
      channels_per_group, false, true);

  arma::fmat multi_kernel_channel(group_kernel->raw_ptr(), kernel_hw,
                                  channels_per_group, false, true);
  return multi_kernel_channel * (multi_input_channel.t());
}

void DeconvolutionLayer::DeconvCol2ImBias(
    const arma::fmat& gemm_result, sftensor output_tensor, uint32_t input_h,
    uint32_t input_w, uint32_t group, uint32_t kernel_index,
    uint32_t kernel_count_group, uint32_t kernel_h, uint32_t kernel_w,
    uint32_t output_h, uint32_t output_w) const {
  CHECK(!gemm_result.empty());
  CHECK(input_h > 0 && input_w > 0);
  CHECK(output_tensor != nullptr && !output_tensor->empty());

  arma::fmat output_padding(output_h + 2 * padding_h_,
                            output_w + 2 * padding_w_);

  uint32_t slide_count_w = input_w;
  uint32_t slide_count_h = input_h;

  for (uint32_t index = 0; index < slide_count_w * slide_count_h; ++index) {
    uint32_t x = index / slide_count_h;
    uint32_t y = index % slide_count_h;
    const uint32_t offset_x = x * stride_w_;
    const uint32_t offset_y = y * stride_h_;
    arma::fmat gemm_column((float*)gemm_result.colptr(index), kernel_h,
                           kernel_w, false, true);

    uint32_t gemm_rows = gemm_column.n_rows;
    uint32_t gemm_cols = gemm_column.n_cols;

    for (uint32_t col = 0; col < gemm_cols; ++col) {
      float* gemm_ptr = gemm_column.colptr(col);
      float* output_ptr = output_padding.colptr(offset_x + col);
      for (uint32_t row = 0; row < gemm_rows; ++row) {
        *(output_ptr + offset_y + row) += *(gemm_ptr + row);
      }
    }
  }

  kernel_index = kernel_index + group * kernel_count_group;
  arma::fmat output(output_tensor->matrix_raw_ptr(kernel_index), output_h,
                    output_w, false, true);

  output =
      output_padding.submat(padding_h_, padding_w_, output_h + padding_h_ - 1,
                            output_w + padding_w_ - 1);

  AddBias(output, kernel_index);
}

StatusCode BaseConvolutionLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the convolution layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the convolution layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the convolution "
                  "layer do not match";
    return StatusCode::kInferInOutSizeMismatch;
  }

  if (weights_.empty()) {
    LOG(ERROR) << "The number of kernel matrix in the convolution layer should "
                  "be greater than zero";
    return StatusCode::kInferParameterError;
  }

  if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
    LOG(ERROR) << "The number of kernel matrix and bias matrix do not match";
    return StatusCode::kInferParameterError;
  }

  if (!stride_h_ || !stride_h_) {
    LOG(ERROR) << "The stride in the convolution layer should be greater "
                  "than zero";
    return StatusCode::kInferParameterError;
  }

  if (!dilation_h_ || !dilation_w_) {
    LOG(ERROR) << "The dilation in the convolution layer should be greater "
                  "than zero";
    return StatusCode::kInferParameterError;
  }

  if (!groups_) {
    LOG(ERROR) << "The group number in the convolution layer should be "
                  "greater than zero ";
    return StatusCode::kInferParameterError;
  }

  if (conv_type_ == ConvType::OpConv) {
    if (output_padding_h_ != 0 || output_padding_w_ != 0) {
      LOG(ERROR)
          << "The output padding in the convolution layer should be zero ";
      return StatusCode::kInferParameterError;
    }
  }

  const uint32_t kernel_count = this->weights_.size();
  const uint32_t kernel_channel = this->weights_.at(0)->channels();

  uint32_t kernel_h = this->weights_.at(0)->rows();
  uint32_t kernel_w = this->weights_.at(0)->cols();
  CHECK(kernel_h > 0 && kernel_w > 0 && kernel_channel > 0)
      << "The size of kernel matrix in the convolution layer should be greater "
         "than zero";

  for (uint32_t k = 0; k < kernel_count; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_channel);
  }

  const uint32_t batch_size = inputs.size();
  const uint32_t kernel_count_group = kernel_count / groups_;

  if (kernel_matrix_arr_.empty()) {
    this->InitIm2ColWeight();
  }

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the convolution layer has an empty  "
           "tensor "
        << i << " th";

    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();
    const uint32_t input_c = input->channels();
    CHECK(input_h > 0 && input_w > 0);

    const auto [output_h, output_w] =
        ComputeOutputSize(input_h, input_w, kernel_h, kernel_w);
    CHECK(output_h > 0 && output_w > 0)
        << "The size of the output tensor should be greater than zero " << i
        << " th";

    std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
    if (output_tensor == nullptr || output_tensor->empty()) {
      output_tensor =
          std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
      outputs.at(i) = output_tensor;
    }

    CHECK(output_tensor->rows() == output_h &&
          output_tensor->cols() == output_w &&
          output_tensor->channels() == kernel_count)
        << "The output tensor array in the convolution layer has an "
           "incorrectly sized tensor "
        << i << "th";

#pragma omp parallel for if (groups_ > 1)
    for (uint32_t group = 0; group < groups_; ++group) {
      if (groups_ != 1) {
        CHECK(kernel_count % groups_ == 0);
        CHECK(input_c % groups_ == 0);
      }
      const uint32_t channels_per_group = input_c / groups_;
      CHECK(channels_per_group == kernel_channel)
          << "The number of channel for the kernel "
             "matrix and input tensor do not match";
      ComputeOutput(input, output_tensor, kernel_h, kernel_w,
                    kernel_count_group, input_h, input_w, channels_per_group,
                    output_h, output_w, group);
    }
  }
  return StatusCode::kSuccess;
}

BaseConvolutionLayer::BaseConvolutionLayer(
    ConvType conv_type, uint32_t output_channel, uint32_t in_channel,
    uint32_t kernel_h, uint32_t kernel_w, uint32_t padding_h,
    uint32_t padding_w, uint32_t stride_h, uint32_t stride_w, uint32_t groups,
    bool use_bias, uint32_t output_padding_h, uint32_t output_padding_w,
    uint32_t dilation_h, uint32_t dilation_w)
    : ParamLayer("convolution"),
      conv_type_(conv_type),
      use_bias_(use_bias),
      groups_(groups),
      padding_h_(padding_h),
      padding_w_(padding_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      output_padding_h_(output_padding_h),
      output_padding_w_(output_padding_w),
      dilation_h_(dilation_h),
      dilation_w_(dilation_w) {
  if (groups != 1) {
    in_channel /= groups;
  }

  CHECK_GE(groups_, 1);
  CHECK_GT(stride_h_, 0);
  CHECK_GT(stride_w_, 0);
  CHECK_GT(dilation_h_, 0);
  CHECK_GT(dilation_w_, 0);

  if (conv_type_ == ConvType::OpConv) {
    CHECK_EQ(output_padding_h_, 0);
    CHECK_EQ(output_padding_w_, 0);
  } else if (conv_type_ == ConvType::OpDeconv) {
    if (dilation_h > 1)
      kernel_h = (kernel_h - 1) * (dilation_h_ - 1) + kernel_h;
    if (dilation_w > 1)
      kernel_w = (kernel_w - 1) * (dilation_w_ - 1) + kernel_w;
  } else {
    LOG(FATAL) << "Unsupported convolution type: " << int(conv_type_);
  }

  CHECK_GT(kernel_h, 0);
  CHECK_GT(kernel_w, 0);
  this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
  if (use_bias_) {
    this->InitBiasParam(output_channel, 1, 1, 1);
  }
}

void BaseConvolutionLayer::AddBias(arma::fmat& output,
                                   uint32_t bias_index) const {
  if (!this->bias_.empty() && this->use_bias_) {
    std::shared_ptr<Tensor<float>> bias;
    bias = this->bias_.at(bias_index);
    if (bias != nullptr && !bias->empty()) {
      float bias_value = bias->index(0);
      output += bias_value;
    } else {
      LOG(FATAL) << "Bias tensor is empty or nullptr";
    }
  }
}

void ConvolutionLayer::InitIm2ColWeight() {
  const uint32_t kernel_count = this->weights_.size();
  CHECK(kernel_count > 0) << "kernel count must greater than zero";
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t row_len = kernel_h * kernel_w;
  CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
      << "The size of kernel matrix should be greater than zero";

  for (uint32_t k = 0; k < kernel_count; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_c);
  }

  std::vector<arma::frowvec> kernel_matrix_arr(kernel_count);
  arma::frowvec kernel_matrix_c(row_len * kernel_c);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
      memcpy(kernel_matrix_c.memptr() + row_len * ic,
             kernel->matrix_raw_ptr(ic), row_len * sizeof(float));
    }
    kernel_matrix_arr.at(k) = kernel_matrix_c;
  }

  if (!kernel_matrix_arr.empty()) {
    if (groups_ == 1) {
      CHECK(kernel_matrix_arr.size() == kernel_count / groups_)
          << "The number of kernel matrix and kernel_count_group do not match";
    } else {
      CHECK(kernel_matrix_arr.size() == kernel_count)
          << "The number of kernel matrix and kernel_count do not match";
    }
  }

  this->kernel_matrix_arr_ = std::move(kernel_matrix_arr);
}

void ConvolutionLayer::ComputeOutput(sftensor input, sftensor output_tensor,
                                     uint32_t kernel_h, uint32_t kernel_w,
                                     uint32_t kernel_count_group,
                                     uint32_t input_h, uint32_t input_w,
                                     uint32_t channels_per_group,
                                     uint32_t output_h, uint32_t output_w,
                                     uint32_t group) const {
  const arma::fmat& input_matrix = ConvIm2Col(
      input, kernel_h, kernel_w, input_h, input_w, channels_per_group, output_h,
      output_w, group, kernel_h * kernel_w, output_h * output_w);
#pragma omp parallel for
  for (uint32_t k = 0; k < kernel_count_group; ++k) {
    ConvGemmBias(input_matrix, output_tensor, group, k, kernel_count_group,
                 output_h, output_w);
  }
}

arma::fmat ConvolutionLayer::ConvIm2Col(sftensor input, uint32_t kernel_h,
                                        uint32_t kernel_w, uint32_t input_h,
                                        uint32_t input_w,
                                        uint32_t channels_per_group,
                                        uint32_t output_h, uint32_t output_w,
                                        uint32_t group, uint32_t row_len,
                                        uint32_t col_len) const {
  CHECK(input && !input->empty());
  arma::fmat input_matrix(channels_per_group * row_len, col_len);

  const float padding_value = 0.f;
#pragma omp parallel for
  for (uint32_t ic = 0; ic < channels_per_group; ++ic) {
    float* input_channel_ptr =
        input->matrix_raw_ptr(ic + group * channels_per_group);
    uint32_t current_col = 0;
    uint32_t channel_row = ic * row_len;
    for (uint32_t w = 0, input_col = 0; w < output_w; ++w) {
      for (uint32_t r = 0, input_row = 0; r < output_h; ++r) {
        float* input_matrix_ptr =
            input_matrix.colptr(current_col) + channel_row;
        current_col += 1;
        for (uint32_t kw = 0; kw < kernel_w * dilation_w_; kw += dilation_w_) {
          const uint32_t region_w = input_h * (input_col + kw - padding_w_);
          for (uint32_t kh = 0; kh < kernel_h * dilation_h_;
               kh += dilation_h_) {
            if ((kh + input_row >= padding_h_ &&
                 kw + input_col >= padding_w_) &&
                (kh + input_row < input_h + padding_h_ &&
                 kw + input_col < input_w + padding_w_)) {
              float* region_ptr =
                  input_channel_ptr + region_w + (input_row + kh - padding_h_);
              *input_matrix_ptr = *region_ptr;
            } else {
              *input_matrix_ptr = padding_value;  // only support zero mode
            }
            input_matrix_ptr += 1;
          }
        }
        input_row += stride_h_;
      }
      input_col += stride_w_;
    }
  }
  return input_matrix;
}

void ConvolutionLayer::ConvGemmBias(const arma::fmat& input_matrix,
                                    sftensor output_tensor, uint32_t group,
                                    uint32_t kernel_index,
                                    uint32_t kernel_count_group,
                                    uint32_t output_h,
                                    uint32_t output_w) const {
  CHECK(!input_matrix.empty());
  CHECK(output_tensor && !output_tensor->empty());

  kernel_index = kernel_index + group * kernel_count_group;
  const arma::frowvec& kernel = this->kernel_matrix_arr_.at(kernel_index);

  arma::fmat output(output_tensor->matrix_raw_ptr(kernel_index), output_h,
                    output_w, false, true);
  output = kernel * input_matrix;

  AddBias(output, kernel_index);
}

std::pair<uint32_t, uint32_t> ConvolutionLayer::ComputeOutputSize(
    const uint32_t input_h, const uint32_t input_w, const uint32_t kernel_h,
    const uint32_t kernel_w) const {
  uint32_t output_h = 0;
  uint32_t output_w = 0;

  CHECK_GT(kernel_h, 0);
  CHECK_GT(kernel_w, 0);

  output_h = (input_h + 2 * padding_h_ - dilation_h_ * (kernel_h - 1) - 1) /
                 stride_h_ +
             1;
  output_w = (input_w + 2 * padding_w_ - dilation_w_ * (kernel_w - 1) - 1) /
                 stride_w_ +
             1;
  return {output_h, output_w};
}

StatusCode BaseConvolutionLayer::CreateInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer<float>>& conv_layer) {
  CHECK(op != nullptr) << "Convolution operator is nullptr";
  const std::map<std::string, std::shared_ptr<RuntimeParameter>>& params =
      op->params;

  if (params.find("dilation") == params.end()) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return StatusCode::kParameterMissing;
  }

  auto dilation_param = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("dilation"));

  if (dilation_param == nullptr || dilation_param->value.size() != 2) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return StatusCode::kParameterMissing;
  }

  const uint32_t dilation_h = dilation_param->value.at(0);
  const uint32_t dilation_w = dilation_param->value.at(1);

  if (params.find("in_channels") == params.end()) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return StatusCode::kParameterMissing;
  }
  auto in_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("in_channels"));
  if (!in_channel) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return StatusCode::kParameterMissing;
  }

  if (params.find("out_channels") == params.end()) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return StatusCode::kParameterMissing;
  }

  auto out_channel =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("out_channels"));
  if (!out_channel) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return StatusCode::kParameterMissing;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return StatusCode::kParameterMissing;
  }

  auto padding =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return StatusCode::kParameterMissing;
  }

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the bias parameter";
    return StatusCode::kParameterMissing;
  }
  auto use_bias =
      std::dynamic_pointer_cast<RuntimeParameterBool>(params.at("bias"));
  if (!use_bias) {
    LOG(ERROR) << "Can not find the bias parameter";
    return StatusCode::kParameterMissing;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return StatusCode::kParameterMissing;
  }
  auto stride =
      std::dynamic_pointer_cast<RuntimeParameterIntArray>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return StatusCode::kParameterMissing;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return StatusCode::kParameterMissing;
  }
  auto kernel = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("kernel_size"));
  if (!kernel) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return StatusCode::kParameterMissing;
  }

  if (op->type == "nn.Conv2d") {
    if (params.find("padding_mode") != params.end()) {
      auto padding_mode = std::dynamic_pointer_cast<RuntimeParameterString>(
          params.at("padding_mode"));
      if (padding_mode == nullptr) {
        LOG(ERROR) << "Can not find the padding parameter";
        return StatusCode::kParameterMissing;
      } else {
        const std::string& padding_mode_str = padding_mode->value;
        if (padding_mode_str != "zeros") {
          LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
          return StatusCode::kParameterMissing;
        }
      }
    } else {
      LOG(ERROR) << "Can not find the padding parameter";
      return StatusCode::kParameterMissing;
    }
  }

  if (params.find("groups") == params.end()) {
    LOG(ERROR) << "Can not find the groups parameter";
    return StatusCode::kParameterMissing;
  }

  auto groups =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("groups"));
  if (!groups) {
    LOG(ERROR) << "Can not find the groups parameter";
    return StatusCode::kParameterMissing;
  }

  const uint32_t dims = 2;
  const std::vector<int32_t>& kernels = kernel->value;
  const std::vector<int32_t>& paddings = padding->value;
  const std::vector<int32_t>& strides = stride->value;
  if (paddings.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return StatusCode::kParameterMissing;
  }

  if (strides.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return StatusCode::kParameterMissing;
  }

  if (kernels.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return StatusCode::kParameterMissing;
  }

  uint32_t output_padding_h = 0;
  uint32_t output_padding_w = 0;
  if (op->type == "nn.ConvTranspose2d") {
    if (params.find("output_padding") != params.end()) {
      auto output_padding_arr =
          std::dynamic_pointer_cast<RuntimeParameterIntArray>(
              params.at("output_padding"));
      if (!output_padding_arr) {
        return StatusCode::kParameterMissing;
      } else {
        if (output_padding_arr->value.size() != 2) {
          return StatusCode::kParameterMissing;
        }
        output_padding_h = output_padding_arr->value.at(0);
        output_padding_w = output_padding_arr->value.at(1);
      }
    } else {
      return StatusCode::kParameterMissing;
    }
  }

  ConvType conv_type = ConvType::OpConv;
  if (op->type == "nn.Conv2d") {
    conv_type = ConvType::OpConv;
  } else if (op->type == "nn.ConvTranspose2d") {
    conv_type = ConvType::OpDeconv;
  } else {
    LOG(FATAL) << "Unknown convolution type: " << op->type;
  }

  if (conv_type == ConvType::OpConv) {
    conv_layer = std::make_shared<ConvolutionLayer>(
        out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
        paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
        groups->value, use_bias->value, output_padding_h, output_padding_w,
        dilation_h, dilation_w);
  } else {
    conv_layer = std::make_shared<DeconvolutionLayer>(
        out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
        paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
        groups->value, use_bias->value, output_padding_h, output_padding_w,
        dilation_h, dilation_w);
  }

  // load weights
  const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attrs =
      op->attribute;
  if (use_bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Can not find the bias attribute";
      return StatusCode::kAttributeMissing;
    }
    const auto& bias = attrs.at("bias");
    const std::vector<int32_t>& bias_shape = bias->shape;
    if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
      LOG(ERROR) << "The attribute of bias shape is wrong";
      return StatusCode::kAttributeMissing;
    }

    const std::vector<float>& bias_values = bias->get<float>();
    conv_layer->set_bias(bias_values);
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the weight attribute";
    return StatusCode::kAttributeMissing;
  }

  const auto& weight = attrs.at("weight");
  const std::vector<int32_t>& weight_shape = weight->shape;
  if (weight_shape.empty()) {
    LOG(ERROR) << "The attribute of weight shape is wrong";
    return StatusCode::kAttributeMissing;
  }

  const std::vector<float>& weight_values = weight->get<float>();
  conv_layer->set_weights(weight_values);

  auto conv_layer_derived =
      std::dynamic_pointer_cast<BaseConvolutionLayer>(conv_layer);
  CHECK(conv_layer_derived != nullptr);
  conv_layer_derived->InitIm2ColWeight();

  return StatusCode::kSuccess;
}

LayerRegistererWrapper kConvCreateInstance(
    "nn.Conv2d", BaseConvolutionLayer::CreateInstance);

LayerRegistererWrapper kDeConvCreateInstance(
    "nn.ConvTranspose2d", BaseConvolutionLayer::CreateInstance);
}  // namespace kuiper_infer