//
// Created by fss on 22-11-13.
//

#include "convolution.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

namespace kuiper_infer {
ConvolutionLayer::ConvolutionLayer(uint32_t output_channel, uint32_t in_channel,
                                   uint32_t kernel_h, uint32_t kernel_w,
                                   uint32_t padding_h, uint32_t padding_w,
                                   uint32_t stride_h, uint32_t stride_w,
                                   uint32_t groups, bool use_bias)
    : ParamLayer("Convolution"),
      use_bias_(use_bias),
      groups_(groups),
      padding_h_(padding_h),
      padding_w_(padding_w),
      stride_h_(stride_h),
      stride_w_(stride_w) {
  if (groups != 1) {
    in_channel /= groups;
  }
  this->InitWeightParam(output_channel, in_channel, kernel_h, kernel_w);
  if (use_bias_) {
    this->InitBiasParam(output_channel, 1, 1, 1);
  }
}

InferStatus ConvolutionLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of convolution layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  if (weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
    LOG(ERROR) << "The size of the weight and bias is not adapting";
    return InferStatus::kInferFailedBiasParameterError;
  }

  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be "
                  "greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }

  const uint32_t kernel_count = this->weights_.size();
  CHECK(kernel_count > 0) << "kernel count must greater than zero";
  const uint32_t kernel_h = this->weights_.at(0)->rows();
  const uint32_t kernel_w = this->weights_.at(0)->cols();
  const uint32_t kernel_c = this->weights_.at(0)->channels();
  const uint32_t row_len = kernel_h * kernel_w;
  CHECK(kernel_h > 0 && kernel_w > 0 && kernel_c > 0)
      << "The size of kernel size is less than zero";

  for (uint32_t k = 0; k < kernel_count; ++k) {
    const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
    CHECK(kernel->rows() == kernel_h);
    CHECK(kernel->cols() == kernel_w);
    CHECK(kernel->channels() == kernel_c);
  }
  const uint32_t kernel_count_group = kernel_count / groups_;
  const uint32_t batch_size = inputs.size();

  std::vector<arma::frowvec> kernel_matrix_arr(kernel_count_group);
  if (groups_ == 1) {
    arma::frowvec kernel_matrix_c(row_len * kernel_c);
    for (uint32_t k = 0; k < kernel_count_group; ++k) {
      const std::shared_ptr<Tensor<float>>& kernel = this->weights_.at(k);
      for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
        memcpy(kernel_matrix_c.memptr() + row_len * ic,
               kernel->slice(ic).memptr(), row_len * sizeof(float));
      }
      kernel_matrix_arr.at(k) = kernel_matrix_c;
    }
  }
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input feature map of conv layer is empty";

    std::shared_ptr<Tensor<float>> input_;
    if (padding_h_ > 0 || padding_w_ > 0) {
      input_ = TensorPadding(
          input, {padding_h_, padding_h_, padding_w_, padding_w_}, 0);
    } else {
      input_ = input;
    }

    const uint32_t input_w = input_->cols();
    const uint32_t input_h = input_->rows();
    const uint32_t input_c = input_->channels();

    const uint32_t output_h =
        uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
    const uint32_t output_w =
        uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));
    CHECK(output_h > 0 && output_w > 0)
        << "The size of the output feature map is less than zero";

    if (groups_ != 1) {
      CHECK(kernel_count % groups_ == 0);
      CHECK(input_c % groups_ == 0);
    }

    uint32_t col_len = output_h * output_w;
    CHECK(col_len > 0) << "The col len of the input matrix is less than zero";

    uint32_t input_c_group = input_c / groups_;
    CHECK(input_c_group == kernel_c)
        << "The channel of the kernel and input feature do not equal";

    for (uint32_t g = 0; g < groups_; ++g) {
      std::vector<arma::fmat> kernel_matrix_arr_group(kernel_count_group);
      if (groups_ != 1) {
        arma::fmat kernel_matrix_c(1, row_len * kernel_c);
        for (uint32_t k = 0; k < kernel_count_group; ++k) {
          const std::shared_ptr<Tensor<float>>& kernel =
              this->weights_.at(k + g * kernel_count_group);
          for (uint32_t ic = 0; ic < kernel->channels(); ++ic) {
            memcpy(kernel_matrix_c.memptr() + row_len * ic,
                   kernel->slice(ic).memptr(), row_len * sizeof(float));
          }
          kernel_matrix_arr_group.at(k) = kernel_matrix_c;
        }
      }

      arma::fmat input_matrix(input_c_group * row_len, col_len);
      for (uint32_t ic = 0; ic < input_c_group; ++ic) {
        const arma::fmat& input_channel = input_->slice(ic + g * input_c_group);
        int current_col = 0;
        for (uint32_t w = 0; w < input_w - kernel_w + 1; w += stride_w_) {
          for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {
            float* input_matrix_c_ptr =
                input_matrix.colptr(current_col) + ic * row_len;
            current_col += 1;

            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
              const float* region_ptr = input_channel.colptr(w + kw) + r;
              memcpy(input_matrix_c_ptr, region_ptr, kernel_h * sizeof(float));
              input_matrix_c_ptr += kernel_h;
            }
          }
        }
      }

      std::shared_ptr<Tensor<float>> output_tensor = outputs.at(i);
      if (output_tensor == nullptr || output_tensor->empty()) {
        output_tensor =
            std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
        outputs.at(i) = output_tensor;
      }

      CHECK(output_tensor->rows() == output_h &&
            output_tensor->cols() == output_w &&
            output_tensor->channels() == kernel_count)
          << "The output size of convolution is error";
#pragma omp parallel for schedule(dynamic)
      for (uint32_t k = 0; k < kernel_count_group; ++k) {
        arma::fmat output(
            output_tensor->slice(k + g * kernel_count_group).memptr(), output_h,
            output_w, false, true);
        if (groups_ == 1) {
          output = kernel_matrix_arr.at(k) * input_matrix;
        } else {
          output = kernel_matrix_arr_group.at(k) * input_matrix;
        }
        std::shared_ptr<Tensor<float>> bias;
        if (!this->bias_.empty() && this->use_bias_) {
          bias = this->bias_.at(k);
        }
        CHECK(output.size() == output_h * output_w);
        if (bias != nullptr && !bias->empty()) {
          float bias_value = bias->index(0);
          output += bias_value;
        }
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ConvolutionLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& conv_layer) {
  CHECK(op != nullptr) << "Convolution operator is nullptr";
  const std::map<std::string, RuntimeParameter*>& params = op->params;

  if (params.find("dilation") == params.end()) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return ParseParameterAttrStatus::kParameterMissingDilation;
  }

  const auto& dilation_param =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("dilation"));

  if (dilation_param == nullptr || dilation_param->value.size() != 2) {
    LOG(ERROR) << "Can not find the dilation parameter";
    return ParseParameterAttrStatus::kParameterMissingDilation;
  }

  CHECK(dilation_param->value.at(0) != 1 || dilation_param->value.at(1))
      << "Only support dilation value equals to one!";

  if (params.find("in_channels") == params.end()) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }
  const auto& in_channel =
      dynamic_cast<RuntimeParameterInt*>(params.at("in_channels"));
  if (!in_channel) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }

  if (params.find("out_channels") == params.end()) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  const auto& out_channel =
      dynamic_cast<RuntimeParameterInt*>(params.at("out_channels"));
  if (!out_channel) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  const auto& padding =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  const auto& use_bias = dynamic_cast<RuntimeParameterBool*>(params.at("bias"));
  if (!use_bias) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }
  const auto& stride =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  const auto& kernel =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("kernel_size"));
  if (!kernel) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  if (params.find("padding_mode") != params.end()) {
    const auto& padding_mode =
        dynamic_cast<RuntimeParameterString*>(params.at("padding_mode"));
    if (padding_mode == nullptr) {
      LOG(ERROR) << "Can not find the padding parameter";
      return ParseParameterAttrStatus::kParameterMissingPaddingMode;
    } else {
      const std::string& padding_mode_str = padding_mode->value;
      if (padding_mode_str != "zeros") {
        LOG(ERROR) << "Padding mode unsupported: " << padding_mode_str;
        return ParseParameterAttrStatus::kParameterMissingPaddingMode;
      }
    }
  } else {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPaddingMode;
  }

  const auto& groups = dynamic_cast<RuntimeParameterInt*>(params.at("groups"));
  if (!groups) {
    LOG(ERROR) << "Can not find the groups parameter";
    return ParseParameterAttrStatus::kParameterMissingGroups;
  }

  const uint32_t dims = 2;
  const std::vector<int>& kernels = kernel->value;
  const std::vector<int>& paddings = padding->value;
  const std::vector<int>& strides = stride->value;
  if (paddings.size() != dims) {
    LOG(ERROR) << "Can not find the right padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (strides.size() != dims) {
    LOG(ERROR) << "Can not find the right stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (kernels.size() != dims) {
    LOG(ERROR) << "Can not find the right kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  // kernel的方向是倒置的
  conv_layer = std::make_shared<ConvolutionLayer>(
      out_channel->value, in_channel->value, kernels.at(0), kernels.at(1),
      paddings.at(0), paddings.at(1), strides.at(0), strides.at(1),
      groups->value, use_bias->value);

  // load weights
  const std::map<std::string, std::shared_ptr<RuntimeAttribute>>& attrs =
      op->attribute;
  if (use_bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Can not find the bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto& bias = attrs.at("bias");
    const std::vector<int>& bias_shape = bias->shape;
    if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
      LOG(ERROR) << "The attribute of bias shape is wrong";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }

    const std::vector<float>& bias_values = bias->get<float>();
    conv_layer->set_bias(bias_values);
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the weight attribute";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const auto& weight = attrs.at("weight");
  const std::vector<int>& weight_shape = weight->shape;
  if (weight_shape.empty()) {
    LOG(ERROR) << "The attribute of weight shape is wrong";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const std::vector<float>& weight_values = weight->get<float>();
  conv_layer->set_weights(weight_values);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kConvGetInstance("nn.Conv2d",
                                        ConvolutionLayer::GetInstance);
}  // namespace kuiper_infer
