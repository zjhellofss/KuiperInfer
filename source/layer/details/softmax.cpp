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

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(input != nullptr && !input->empty()) << "The input feature map for softmax layer is empty";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      LOG(ERROR) << "The output size of softmax is empty" << "The output size of softmax is error";
      outputs.at(i) = output;
    }

    CHECK(input->shapes() == output->shapes()) << "The output size of softmax is error";

    const arma::fcube &input_data = input->data();
    arma::fcube &output_data = output->data();
    const float max = input_data.max();
    const float sum = arma::accu(arma::exp(input_data - max));
    const float offset = max + logf(sum);

    output_data = arma::exp(input_data - offset);
  }
  return InferStatus::kInferSuccess;
}

}