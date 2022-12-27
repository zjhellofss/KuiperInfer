//
// Created by fss on 22-12-27.
//
#include "tuple_construct.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
TupleConstructLayer::TupleConstructLayer() : Layer("tuple_construct") {

}

InferStatus TupleConstructLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus TupleConstructLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                          std::shared_ptr<Layer> &tuple_layer) {
  tuple_layer = std::make_shared<TupleConstructLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kTupleGetInstance("prim::TupleConstruct", TupleConstructLayer::GetInstance);

}

