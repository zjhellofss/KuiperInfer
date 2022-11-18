//
// Created by fss on 22-11-18.
//

#include "add.hpp"
namespace kuiper_infer {
InferStatus AddLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs1,
                              const std::vector<std::shared_ptr<Tensor>> &inputs2,
                              std::vector<std::shared_ptr<Tensor>> &outputs) {
  if (inputs1.empty() || inputs2.empty()) {
    LOG(ERROR) << "The input feature map of add layer is null";
    return InferStatus::kInferFailedInputEmpty;
  }
  CHECK(inputs2.size() == inputs2.size());
  const uint32_t size = inputs1.size();
  for (uint32_t i = 0; i < size; ++i) {
    const auto &input1 = inputs1.at(i);
    const auto &input2 = inputs2.at(i);
    input1->Add(input2);
    outputs.push_back(input1);
  }
}
}
