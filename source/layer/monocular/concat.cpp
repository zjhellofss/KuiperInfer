//
// Created by fss on 22-11-16.
//

#include "concat.hpp"
#include "parser/runtime_ir.hpp"
#include "../binocular/layer_factory.hpp"

namespace kuiper_infer {

InferStatus ConcatLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs1,
                                 const std::vector<std::shared_ptr<Tensor>> &inputs2,
                                 std::vector<std::shared_ptr<Tensor>> &outputs) {

  if (inputs1.empty() || inputs2.empty()) {
    LOG(ERROR) << "The input feature map of concat layer is null";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (dim_ != 1) {
    return InferStatus::kInferFailedDimensionParameterError;
  }

  CHECK(inputs1.size() == inputs2.size());
  const uint32_t size = inputs1.size();
  for (uint32_t i = 0; i < size; ++i) {
    const auto &input1 = inputs1.at(i);
    const auto &input2 = inputs2.at(i);
    input1->Concat(input2);
    outputs.push_back(input1);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ConcatLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                  std::shared_ptr<Layer> &concat_layer) {
  const std::map<std::string, RuntimeParameter *> params = op->params;

  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  const auto &dim = dynamic_cast<RuntimeParameterInt *>(params.at("dim"));
  if (!dim) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  concat_layer = std::make_shared<ConcatLayer>(dim->value);
  return ParseParameterAttrStatus::kParameterParseSuccess;
}

LayerRegistererWrapper kConcatGetInstance("torch.cat", ConcatLayer::GetInstance);

}

