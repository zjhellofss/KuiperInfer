//
// Created by fss on 22-11-13.
//

#include "softmax.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
SoftmaxLayer::SoftmaxLayer() : Layer("Softmax") {

}

InferStatus SoftmaxLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of softmax layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of softmax layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    const std::shared_ptr<Tensor<float>> &output = input->Clone();

    const arma::fcube &input_data = input->data();
    arma::fcube &output_data = output->data();

    const double max = input_data.max();
    const double sum = arma::accu(arma::exp(input_data - max));
    const double offset = max + log(sum);
    output_data = arma::exp(input_data - offset);
    outputs.push_back(output);
  }
  return InferStatus::kInferSuccess;
}

}