//
// Created by fss on 22-11-12.
//
#include "avgpooling.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

AveragePoolingLayer::AveragePoolingLayer(uint32_t kernel_size, uint32_t padding, uint32_t stride)
    : Layer("AveragePooling"), pooling_size_(kernel_size), padding_(padding), stride_(stride) {

}

InferStatus AveragePoolingLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                         std::vector<std::shared_ptr<Blob>> &outputs) {

  if (inputs.empty()) {
    LOG(ERROR) << "The input is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch = inputs.size();
  const uint32_t pooling_w = pooling_size_;
  const uint32_t pooling_h = pooling_size_;
  if (!stride_) {
    LOG(ERROR) << "The stride parameter is set incorrectly. It must always be greater than 1";
    stride_ = 1;
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Blob> &input = inputs.at(i);
    if (this->padding_ > 0) {
      input->Padding({padding_, padding_, padding_, padding_}, 0);
    }
    const uint32_t input_h = input->rows();
    const uint32_t input_w = input->cols();
    const uint32_t input_c = input->channels();
    const uint32_t output_c = input_c;

    const uint32_t output_h = uint32_t(std::floor((input_h - pooling_h) / stride_ + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - pooling_w) / stride_ + 1));
    if (output_w <= 0 || output_h <= 0) {
      LOG(ERROR) << "Output size is less than 0";
      return InferStatus::kInferFailedOutputEmpty;
    }

    std::shared_ptr<Blob> output_blob = std::make_shared<Blob>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::mat &input_channel = input->at(ic);
      arma::mat &output_channel = output_blob->at(ic);
      for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_) {
        for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_) {
          const arma::mat &region = input_channel.submat(r, c, r + pooling_h - 1, c + pooling_w - 1);
          output_channel.at(int(r / stride_), int(c / stride_)) = arma::mean(arma::mean(region));
        }
      }
    }
    outputs.push_back(output_blob);
  }
  return InferStatus::kInferSuccess;
}
}