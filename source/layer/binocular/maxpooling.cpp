//
// Created by fss on 22-11-18.
//

#include "maxpooling.hpp"
#include "parser/runtime_ir.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {

MaxPoolingLayer::MaxPoolingLayer(uint32_t padding_h, uint32_t padding_w,
                                 uint32_t pooling_size_h, uint32_t pooling_size_w,
                                 uint32_t stride_h, uint32_t stride_w)
    : Layer("MaxPooling"), padding_h_(padding_h), padding_w_(padding_w), pooling_size_h_(pooling_size_h),
      pooling_size_w_(pooling_size_w), stride_h_(stride_h), stride_w_(stride_w) {

}

InferStatus MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of average pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch = inputs.size();
  const uint32_t pooling_w = pooling_size_h_;
  const uint32_t pooling_h = pooling_size_w_;
  if (!stride_h_ || !stride_w_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be greater than 0";
    return InferStatus::kInferFailedStrideParameterError;
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    if (input_data->empty()) {
      LOG(ERROR) << "The input feature map of average pooling layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (!padding_h_ || !padding_w_) {
      input_data->Padding({padding_h_, padding_h_, padding_w_, padding_w_}, 0);
    }
    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t output_c = input_c;

    const uint32_t output_h = uint32_t(std::floor((input_h - pooling_h) / stride_h_ + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - pooling_w) / stride_w_ + 1));
    if (output_w <= 0 || output_h <= 0) {
      LOG(ERROR) << "The size of the output feature map is less than zero";
      return InferStatus::kInferFailedOutputSizeError;
    }

    std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data->at(ic);
      arma::fmat &output_channel = output_data->at(ic);
      for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w_) {
        for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h_) {

          double max_value = std::numeric_limits<double>::min();
          for (uint32_t kw = 0; kw < pooling_w; ++kw) {
            auto *region_ptr_col = input_channel.colptr(kw + c) + r;
            for (uint32_t kh = 0; kh < pooling_h; ++kh) {
              auto *region_ptr = region_ptr_col + kh;
              if (max_value < *region_ptr) {
                max_value = *region_ptr;
              }
            }
          }
          output_channel.at(int(r / stride_h_), int(c / stride_w_)) = max_value;
        }
      }
    }
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus MaxPoolingLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &max_layer) {
  CHECK(op != nullptr) << "MaxPooling get instance failed, operator is null";

  const std::map<std::string, RuntimeParameter *> &params = op->params;
  if (params.find("stride") == params.end()) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
  }

  const auto &stride = dynamic_cast<RuntimeParameterIntArray *>(params.at("stride"));
  if (!stride) {
    LOG(ERROR) << "Can not find the stride parameter";
    return ParseParameterAttrStatus::kParameterMissingStride;
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

  if (params.find("kernel_size") == params.end()) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }

  const auto &kernel_size = dynamic_cast<RuntimeParameterIntArray *>(params.at("kernel_size"));
  if (!kernel_size) {
    LOG(ERROR) << "Can not find the kernel size parameter";
    return ParseParameterAttrStatus::kParameterMissingKernel;
  }
  const auto &padding_values = padding->value;
  const auto &stride_values = stride->value;
  const auto &kernel_values = kernel_size->value;

  CHECK(padding_values.size() == 2);
  CHECK(stride_values.size() == 2);
  CHECK(kernel_values.size() == 2);

  max_layer = std::make_shared<MaxPoolingLayer>(padding_values.at(0), padding_values.at(1),
                                                kernel_values.at(0), kernel_values.at(1),
                                                stride_values.at(0), stride_values.at(1));

  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kMaxPoolingGetInstance("nn.MaxPool2d", MaxPoolingLayer::GetInstance);

}