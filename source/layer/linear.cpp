//
// Created by fss on 22-11-13.
//

#include "linear.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

LinearLayer::LinearLayer(const std::vector<std::shared_ptr<Blob>> &weights,
                         const std::vector<std::shared_ptr<Blob>> &bias) : ParamLayer("Linear", weights, bias) {

}

InferStatus LinearLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                 std::vector<std::shared_ptr<Blob>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (this->weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return InferStatus::kInferFailedWeightsBiasEmpty;
  } else {
    if (!this->bias_.empty() && this->weights_.size() != this->bias_.size()) {
      LOG(ERROR) << "The size of the weight and bias is not adapting";
      return InferStatus::kInferFailedWeightsBiasEmpty;
    }
  }

  if (this->weights_.size() != inputs.size()) {
    return InferStatus::kInferFailedWeightBiasNoAdapting;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Blob> &weight = this->weights_.at(i);
    const std::shared_ptr<Blob> &input = inputs.at(i);
    std::shared_ptr<Blob> bias;

    if (!this->bias_.empty() && i < this->bias_.size()) {
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
        if (bias != nullptr) {
          if (c < bias->channels() && bias->cols() == output_data.n_cols &&
              bias->rows() == output_data.n_rows) {
            output_data += bias->at(c);
          } else {
            LOG(ERROR) << "The size of the bias and input is not adapting";
            return InferStatus::kInferFailedWeightBiasNoAdapting;
          }
        }
      }
    }
    outputs.push_back(output);
  }
  return InferStatus::kInferSuccess;
}

}