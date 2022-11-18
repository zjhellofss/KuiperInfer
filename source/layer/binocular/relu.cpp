//
// Created by fss on 22-11-18.
//
#include "relu.hpp"
namespace kuiper_infer {
InferStatus ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                               std::vector<std::shared_ptr<Tensor>> &outputs) {
  if (inputs.empty()) {
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor> &input = inputs.at(i);
    if (input->empty()) {
      return InferStatus::kInferFailedInputEmpty;
    }
    
    input->Transform(0.);
  }

  return InferStatus::kInferSuccess;
}
}

