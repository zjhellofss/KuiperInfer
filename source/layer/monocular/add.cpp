//
// Created by fss on 22-11-18.
//

#include "add.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
AddLayer::AddLayer() : Layer("Add") {

}

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
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus AddLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                               std::shared_ptr<Layer> &add_layer) {
  CHECK(op != nullptr) << "Add operator is empty";
  add_layer = std::make_shared<AddLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kAddGetInstance("pnnx.Expression", AddLayer::GetInstance);
}
