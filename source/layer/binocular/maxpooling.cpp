//
// Created by fss on 22-11-18.
//

#include "maxpooling.hpp"
namespace kuiper_infer {

MaxPoolingLayer::MaxPoolingLayer(uint32_t pooling_size_h, uint32_t pooling_size_w,
                                 uint32_t stride_h, uint32_t stride_w)
    : Layer("MaxPooling"), pooling_size_h_(pooling_size_h),
      pooling_size_w_(pooling_size_w), stride_h_(stride_h), stride_w_(stride_w) {

}

InferStatus MaxPoolingLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                     std::vector<std::shared_ptr<Tensor>> &outputs) {
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
    const std::shared_ptr<Tensor> &input_data = inputs.at(i);
    if (input_data->empty()) {
      LOG(ERROR) << "The input feature map of average pooling layer is empty";
      return InferStatus::kInferFailedInputEmpty;
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

    std::shared_ptr<Tensor> output_data = std::make_shared<Tensor>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::mat &input_channel = input_data->at(ic);
      arma::mat &output_channel = output_data->at(ic);
      for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h_) {
        for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w_) {
          const arma::mat &region = input_channel.submat(r, c, r + pooling_h - 1, c + pooling_w - 1);
          output_channel.at(int(r / stride_h_), int(c / stride_w_)) = arma::max(arma::max(region));
        }
      }
    }
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}

}
