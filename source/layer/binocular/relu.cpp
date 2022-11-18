//
// Created by fss on 22-11-18.
//
#include "relu.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
InferStatus ReluLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                               std::vector<std::shared_ptr<Tensor>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of relu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor> &input = inputs.at(i);
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of relu layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    input->Transform([](double val) {
      return val > 0. ? val : 0.;
    });
  }

  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus ReluLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                std::shared_ptr<Layer> &relu_layer) {
  CHECK(op != nullptr) << "Relu operator is empty";
  relu_layer = std::make_shared<ReluLayer>();
}

LayerRegistererWrapper kFlattenGetInstance("nn.ReLU", ReluLayer::GetInstance);
}

