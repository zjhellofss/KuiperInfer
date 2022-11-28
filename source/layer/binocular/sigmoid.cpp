//
// Created by fss on 22-11-13.
//

#include "sigmoid.hpp"
#include <glog/logging.h>
namespace kuiper_infer {

InferStatus SigmoidLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of sigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  CHECK(inputs.size() == outputs.size());
  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of sigmoid layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    CHECK(output != nullptr);
    CHECK(output->channels() == input->channels() && output->rows() == input->rows()
              && output->cols() == input->cols());

    arma::fcube &output_data_cube = output->data();
    output_data_cube = 1 / (1 + arma::exp(output_data_cube));
  }
  return InferStatus::kInferSuccess;
}
}