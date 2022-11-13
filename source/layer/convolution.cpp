//
// Created by fss on 22-11-13.
//

#include "convolution.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

ConvolutionLayer::ConvolutionLayer(const std::vector<std::shared_ptr<Blob>> &weights,
                                   const std::vector<std::shared_ptr<Blob>> &bias,
                                   uint32_t padding, uint32_t stride) :
    ParamLayer("Convolution", weights, bias), padding_(padding), stride_(stride) {

}

InferStatus ConvolutionLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                      std::vector<std::shared_ptr<Blob>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return InferStatus::kInferFailedWeightsBiasEmpty;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Blob> &input = inputs.at(i);
    if (padding_ > 0) {
      input->Padding({padding_, padding_, padding_, padding_}, 0);
    }

    const uint32_t input_w = input->cols();
    const uint32_t input_h = input->rows();
    const uint32_t input_c = input->channels();

    const uint32_t kernel_count = this->weights_.size();
    std::shared_ptr<Blob> output_data;

    for (uint32_t k = 0; k < kernel_count; ++k) {
      const std::shared_ptr<Blob> &kernel = this->weights_.at(k);
      const uint32_t kernel_h = kernel->rows();
      const uint32_t kernel_w = kernel->cols();

      uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_ + 1));
      uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_ + 1));
      if (output_h <= 0 || output_w <= 0) {
        LOG(ERROR) << "The output is empty";
        return InferStatus::kInferFailedOutputEmpty;
      }

      if (!output_data) {
        output_data = std::make_shared<Blob>(kernel_count, output_h, output_w);
      }

      if (kernel->channels() != input_c) {
        LOG(ERROR) << "The channel of the weight and input is not adapting";
        return InferStatus::kInferFailedInputUnAdapting;
      }

      arma::mat &output_channel = output_data->at(k);
      for (uint32_t ic = 0; ic < input_c; ++ic) {
        const arma::mat &input_channel = input->at(ic);
        const arma::mat &kernel_channel = kernel->at(ic);

        for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_) {
          for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_) {
            const arma::mat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
            output_channel.at(int(r / stride_), int(c / stride_)) += arma::accu(region % kernel_channel);
          }
        }
      }

      std::shared_ptr<Blob> bias;
      if (!this->bias_.empty() && k < this->bias_.size()) {
        bias = this->bias_.at(k);
      }

      if (!bias->empty()) {
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

}