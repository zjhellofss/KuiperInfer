//
// Created by fss on 22-11-13.
//

#include "sigmoid.hpp"
namespace kuiper_infer {

InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                  std::vector<std::shared_ptr<Blob>> &outputs) {
  if (inputs.empty()) {
    return InferStatus::kInferFailedInputEmpty;
  }
  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Blob> &input_data = inputs.at(i);
    std::shared_ptr<Blob> output_data = input_data->Clone();
    arma::cube &output_data_cube = output_data->data();
    output_data_cube = 1 / (1 + arma::exp(output_data_cube));
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}
}