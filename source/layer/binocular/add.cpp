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
  CHECK(size % 2 == 0);
  if (!size) {
    LOG(ERROR) << "The input feature map of add layer is null";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = outputs.size();

  for (uint32_t i = 0; i < batch_size; ++i) {
    outputs.at(i)->Fill(0.f);
  }

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i)->Add(inputs.at(i + batch_size));
    outputs.at(i) = inputs.at(i);
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
