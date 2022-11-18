//
// Created by fss on 22-11-12.
//
#include "avgpooling.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w)
    : Layer("AdaptiveAveragePooling"), output_h_(output_h), output_w_(output_w) {
}

InferStatus AdaptiveAveragePoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                                 std::vector<std::shared_ptr<Tensor>> &outputs) {

  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of average pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (output_w_ <= 0 || output_h_ <= 0) {
    LOG(ERROR) << "The size of the output feature map is less than zero";
    return InferStatus::kInferFailedOutputSizeError;
  }

  // 通过自适应方法中计算stride和pool size
  const uint32_t batch = inputs.size();
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor> &input_data = inputs.at(i);
    if (input_data->empty()) {
      LOG(ERROR) << "The input feature map of average pooling layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }

    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
    const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
    if (!stride_h || !stride_w) {
      LOG(ERROR) << "The stride parameter is set incorrectly. It must always be greater than 0";
      return InferStatus::kInferFailedStrideParameterError;
    }

    const uint32_t kernel_h = input_h - (output_h_ - 1) * stride_h;
    const uint32_t kernel_w = input_w - (output_w_ - 1) * stride_w;
    const uint32_t output_c = input_c;

    std::shared_ptr<Tensor> output_data = std::make_shared<Tensor>(output_c, output_h_, output_w_);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::mat &input_channel = input_data->at(ic);
      arma::mat &output_channel = output_data->at(ic);
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::mat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          output_channel.at(int(r / stride_h), int(c / stride_w)) = arma::mean(arma::mean(region));
        }
      }
    }
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}
}