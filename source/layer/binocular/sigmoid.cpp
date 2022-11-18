//
// Created by fss on 22-11-13.
//

#include "sigmoid.hpp"
#include <glog/logging.h>
namespace kuiper_infer {

InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                  std::vector<std::shared_ptr<Tensor>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of sigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor> &input_data = inputs.at(i);
    if (input_data->empty()) {
      return InferStatus::kInferFailedInputEmpty;
    }
    std::shared_ptr<Tensor> output_data = input_data->Clone();
    arma::cube &output_data_cube = output_data->data();
    output_data_cube = 1 / (1 + arma::exp(output_data_cube));
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}
}