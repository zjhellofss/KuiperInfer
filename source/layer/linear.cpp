//
// Created by fss on 22-11-13.
//

#include "linear.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

LinearLayer::LinearLayer(const std::vector<std::shared_ptr<Blob>> &weights,
                         const std::vector<std::shared_ptr<Blob>> &bias, bool use_bias)
    : ParamLayer("Linear", weights, bias), use_bias_(use_bias) {

}

LinearLayer::LinearLayer(uint32_t batch, uint32_t in_channel, uint32_t in_dim, uint32_t out_dim, bool use_bias)
    : ParamLayer("Linear"), use_bias_(use_bias) {

  for (uint32_t i = 0; i < batch; ++i) {
    std::shared_ptr<Blob> weight = std::make_shared<Blob>(in_channel, in_dim, out_dim);
    this->weights_.push_back(weight);
    if (use_bias_) {
      std::shared_ptr<Blob> bias = std::make_shared<Blob>(in_channel, 1, out_dim);
      this->bias_.push_back(bias);
    }
  }
}

InferStatus LinearLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                 std::vector<std::shared_ptr<Blob>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (this->weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return InferStatus::kInferFailedWeightsOrBiasEmpty;
  } else {
    if (this->use_bias_ && this->weights_.size() != this->bias_.size()) {
      return InferStatus::kInferFailedWeightBiasNoAdapting;
      LOG(ERROR) << "The size of the weight and bias is not adapting";
    }
  }

  if (this->weights_.size() != inputs.size()) {
    LOG(ERROR) << "The size of the weight and input is not adapting";
    return InferStatus::kInferFailedWeightBiasNoAdapting;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Blob> &weight = this->weights_.at(i);
    const std::shared_ptr<Blob> &input = inputs.at(i);
    std::shared_ptr<Blob> bias;

    if (!this->bias_.empty() && this->use_bias_) {
      bias = this->bias_.at(i);
    }

    std::shared_ptr<Blob> output = std::make_shared<Blob>(input->channels(), input->rows(), weight->cols());

    if (input->channels() != weight->channels() || weight->rows() != input->cols()) {
      LOG(ERROR) << "The size of the weight and input is not adapting";
      return InferStatus::kInferFailedWeightBiasNoAdapting;
    } else {
      const uint32_t input_channel = input->channels();
      for (uint32_t c = 0; c < input_channel; ++c) {
        const arma::mat &input_data = input->at(c);
        const arma::mat &weight_data = weight->at(c);
        arma::mat &output_data = output->at(c);
        output_data = input_data * weight_data;
        if (bias != nullptr && this->use_bias_) {
          if (bias->cols() == output_data.n_cols
              && bias->rows() == output_data.n_rows) {
            output_data += bias->at(c);
          } else {
            LOG(ERROR) << "The size of the bias and output is not adapting";
            return InferStatus::kInferFailedOutputSizeWrong;
          }
        }
      }
    }
    outputs.push_back(output);
  }
  return InferStatus::kInferSuccess;
}

}