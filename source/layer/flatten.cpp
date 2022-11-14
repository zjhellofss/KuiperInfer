//
// Created by fss on 22-11-12.
//
#include "flatten.hpp"
#include <glog/logging.h>

namespace kuiper_infer {
FlattenLayer::FlattenLayer() : Layer("Flatten") {

}

InferStatus FlattenLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                  std::vector<std::shared_ptr<Blob>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Blob>& input_data = inputs.at(i);
    const std::shared_ptr<Blob> output_data = input_data->Clone();
    CHECK(!output_data->empty());
    output_data->Flatten();
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}
}
