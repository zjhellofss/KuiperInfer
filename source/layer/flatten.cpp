//
// Created by fss on 22-11-12.
//
#include "flatten.hpp"
#include "parser/runtime_ir.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

FlattenLayer::FlattenLayer() : Layer("Flatten") {

}

InferStatus FlattenLayer::Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                                  std::vector<std::shared_ptr<Blob>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Blob> &input_data = inputs.at(i);
    const std::shared_ptr<Blob> output_data = input_data->Clone();
    CHECK(!output_data->empty());
    output_data->Flatten();
    outputs.push_back(output_data);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus FlattenLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                   std::shared_ptr<Layer> &flatten_layer) {

  const std::map<std::string, RuntimeParameter *> &params = op->params;
  if (params.find("start_dim") == params.end()) {
    LOG(ERROR) << "Can not find the start dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  const auto &start_dim = dynamic_cast<RuntimeParameterInt *>(params.at("start_dim"));
  if (!start_dim) {
    LOG(ERROR) << "Can not find the start dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  if (params.find("end_dim") == params.end()) {
    LOG(ERROR) << "Can not find the end dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  const auto &end_dim = dynamic_cast<RuntimeParameterInt *>(params.at("end_dim"));
  if (!end_dim) {
    LOG(ERROR) << "Can not find the start dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  CHECK(end_dim->value == -1 && start_dim->value == 1);
  flatten_layer = std::make_shared<FlattenLayer>();
  return ParseParameterAttrStatus::kParameterParseSuccess;
}
}
