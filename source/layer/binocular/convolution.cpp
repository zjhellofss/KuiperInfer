//
// Created by fss on 22-11-13.
//

#include "convolution.hpp"
#include "parser/runtime_ir.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

ConvolutionLayer::ConvolutionLayer(uint32_t output_channel, uint32_t in_channel, uint32_t kernel_h,
                                   uint32_t kernel_w, uint32_t padding, uint32_t stride_h, uint32_t stride_w,
                                   bool use_bias)
    : ParamLayer("Convolution"), padding_(padding), stride_h_(stride_h), stride_w_(stride_w), use_bias_(use_bias) {

  for (uint32_t i = 0; i < output_channel; ++i) {
    std::shared_ptr<Tensor> weight = std::make_shared<Tensor>(in_channel, kernel_h, kernel_w);
    this->weights_.push_back(weight);
    if (use_bias_) {
      std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(1, 1, 1);
      this->bias_.push_back(bias);
    }
  }
}

InferStatus ConvolutionLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                      std::vector<std::shared_ptr<Tensor>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of convolution layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (this->use_bias_ && this->bias_.size() != this->weights_.size()) {
    LOG(ERROR) << "The size of the weight and bias is not adapting";
    return InferStatus::kInferFailedBiasParameterError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor> &input = inputs.at(i);
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of convolution layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (padding_ > 0) {
      input->Padding({padding_, padding_, padding_, padding_}, 0);
    }

    const uint32_t input_w = input->cols();
    const uint32_t input_h = input->rows();
    const uint32_t input_c = input->channels();

    const uint32_t kernel_count = this->weights_.size();
    std::shared_ptr<Tensor> output_data;

    for (uint32_t k = 0; k < kernel_count; ++k) {
      const std::shared_ptr<Tensor> &kernel = this->weights_.at(k);
      const uint32_t kernel_h = kernel->rows();
      const uint32_t kernel_w = kernel->cols();

      uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
      uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));
      if (output_h <= 0 || output_w <= 0) {
        LOG(ERROR) << "The size of the output feature map is less than zero";
        return InferStatus::kInferFailedOutputSizeError;
      }

      if (!output_data) {
        output_data = std::make_shared<Tensor>(kernel_count, output_h, output_w);
      }

      if (kernel->channels() != input_c) {
        LOG(ERROR) << "The channel of the weight and input is not adapting";
        return InferStatus::kInferFailedChannelParameterError;
      }

      arma::mat &output_channel = output_data->at(k);
      for (uint32_t ic = 0; ic < input_c; ++ic) {
        const arma::mat &input_channel = input->at(ic);
        const arma::mat &kernel_channel = kernel->at(ic);

        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w_) {
          auto *output_channel_ptr = output_channel.colptr(int(c / stride_h_));
          for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {

            double acc_value = 0.;
            auto *kernel_ptr = const_cast<double *>(kernel_channel.memptr());
            for (uint32_t kw = 0; kw < kernel_w; ++kw) {
              auto *region_ptr_col = input_channel.colptr(kw + c) + r;
              for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                auto *region_ptr = region_ptr_col + kh;
                acc_value += *(region_ptr) * (*kernel_ptr);
                kernel_ptr += 1;
              }
            }
            *(output_channel_ptr + int(r / stride_h_)) += acc_value;
          }
        }
      }

      std::shared_ptr<Tensor> bias;
      if (!this->bias_.empty() && this->use_bias_) {
        bias = this->bias_.at(k);
      }

      if (bias != nullptr) {
        arma::mat bias_mat(output_h, output_w);
        bias_mat.fill(bias->data().front());
        output_channel += bias_mat;
      }
    }
    CHECK(!output_data->empty());
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ConvolutionLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &conv_layer) {
  CHECK(op != nullptr);
  const std::map<std::string, RuntimeParameter *> &params = op->params;

  if (params.find("in_channels") == params.end()) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }

  const auto &in_channel = dynamic_cast<RuntimeParameterInt *>(params.at("in_channels"));
  if (!in_channel) {
    LOG(ERROR) << "Can not find the in channel parameter";
    return ParseParameterAttrStatus::kParameterMissingInChannel;
  }

  if (params.find("out_channels") == params.end()) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  const auto &out_channel = dynamic_cast<RuntimeParameterInt *>(params.at("out_channels"));
  if (!out_channel) {
    LOG(ERROR) << "Can not find the out channel parameter";
    return ParseParameterAttrStatus::kParameterMissingOutChannel;
  }

  if (params.find("padding") == params.end()) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  const auto &padding = dynamic_cast<RuntimeParameterIntArray *>(params.at("padding"));
  if (!padding) {
    LOG(ERROR) << "Can not find the padding parameter";
    return ParseParameterAttrStatus::kParameterMissingPadding;
  }

  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  const auto &use_bias = dynamic_cast<RuntimeParameterBool *>(params.at("bias"));
  if (!use_bias) {
    LOG(ERROR) << "Can not find the bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }
  const auto &stride = dynamic_cast<RuntimeParameterIntArray *>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  const auto &kernel = dynamic_cast<RuntimeParameterIntArray *>(params.at("kernel_size"));
  if (!kernel) {
    LOG(ERROR) << "Can not find the kernel parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  const uint32_t dims = 2;
  const std::vector<int> &kernels = kernel->value;
  const std::vector<int> &paddings = padding->value;
  const std::vector<int> &strides = stride->value;
  CHECK(kernels.size() == dims);
  CHECK(strides.size() == dims);
  CHECK(paddings.size() == dims);

  // kernel的方向是倒置的
  conv_layer = std::make_shared<ConvolutionLayer>(out_channel->value, in_channel->value,
                                                  kernels.at(0), kernels.at(1),
                                                  paddings.at(0), strides.at(0),
                                                  strides.at(1), use_bias->value);

  // load weights
  const std::map<std::string, std::shared_ptr<RuntimeAttribute>> &attrs = op->attribute;
  if (use_bias->value) {
    if (attrs.find("bias") == attrs.end()) {
      LOG(ERROR) << "Can not find the bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto &bias = attrs.at("bias");
    const std::vector<int> &bias_shape = bias->shape;
    if (bias_shape.empty() || bias_shape.at(0) != out_channel->value) {
      LOG(ERROR) << "Bias shape is wrong";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }

    std::vector<double> bias_values = bias->get<double>();
    conv_layer->set_bias(bias_values);
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the weight attribute";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  const auto &weight = attrs.at("weight");
  const std::vector<int> &weight_shape = weight->shape;
  if (weight_shape.empty()) {
    LOG(ERROR) << "Weight shape is empty";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  std::vector<double> weight_values = weight->get<double>();
  conv_layer->set_weights(weight_values);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kConvGetInstance("nn.Conv2d", ConvolutionLayer::GetInstance);

}