//
// Created by fss on 22-11-12.
//
#include "flatten.hpp"
#include "parser/runtime_ir.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

FlattenLayer::FlattenLayer() : Layer("Flatten") {

}

InferStatus FlattenLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  CHECK(inputs.size() == outputs.size());

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    if (input_data->empty()) {
      LOG(ERROR) << "The input feature map of flatten layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    const std::shared_ptr<Tensor<float>> &output_data = outputs.at(i);
    CHECK(output_data != nullptr);
    CHECK(output_data->cols() == input_data->size() && output_data->rows() == 1 && output_data->channels() == 1);

    uint32_t channel = input_data->channels();
    uint32_t rows = input_data->rows();
    uint32_t cols = input_data->cols();
    uint32_t index = 0;

    for (uint32_t c = 0; c < channel; ++c) {
      const arma::fmat &matrix = input_data->data().slice(c);

      for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c_ = 0; c_ < cols; ++c_) {
          output_data->at(0, index, 0) = matrix.at(r, c_);
          index += 1;
        }
      }
    }
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
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kFlattenGetInstance("torch.flatten", FlattenLayer::GetInstance);

}
