//
// Created by fss on 22-11-18.
//

#include "add.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
AddLayer::AddLayer() : Layer("Add") {

}

InferStatus AddLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

  const uint32_t size = inputs.size();
  if (!size) {
    LOG(ERROR) << "The input feature map of add layer is null";
    return InferStatus::kInferFailedInputEmpty;
  }

  const std::shared_ptr<Tensor<float>>& input = inputs.at(0);
  for (uint32_t i = 1; i < size; ++i) {
    input->Add(inputs.at(i));
  }
  outputs.push_back(input);
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
