//
// Created by fss on 22-11-12.
//
#include "view.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

InferStatus ViewLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of view layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The size of input and output feature map is not adapting!";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<ftensor>& input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input feature map of view layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
  }

  if (shapes_.empty()) {
    LOG(ERROR) << "The shape parameter is empty!";
    return InferStatus::kInferFailedShapeParameterError;
  }

  if (shapes_.front() != -1 && shapes_.front() != batch_size) {
    LOG(ERROR) << "The shape parameter is wrong!";
    return InferStatus::kInferFailedShapeParameterError;
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    CHECK(input_data != nullptr && !input_data->empty())
        << "The input feature map of view layer is empty";

    // 检查形状中-1的数量，最多只可以存在一个
    int dynamic_index = -1;
    uint32_t current_size = 1;
    const uint32_t total_size = input_data->size();
    std::vector<uint32_t> shapes;
    for (int j = 1; j < shapes_.size(); ++j) {
      CHECK(shapes_.at(j) == -1 || shapes_.at(j) > 0);
      if (shapes_.at(j) == -1) {
        CHECK(dynamic_index == -1) << "Having two minus one in shape arrays";
        dynamic_index = j;
      } else {
        current_size *= shapes_.at(j);
        shapes.push_back(shapes_.at(j));
      }
    }

    CHECK(dynamic_index == -1 || dynamic_index == shapes_.size() - 1)
        << "Minus one shape is in the wrong axis, only slice the last axis!";
    if (dynamic_index != -1) {
      CHECK(total_size >= current_size);
      shapes.push_back(uint32_t(total_size / current_size));
    }
    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    if (output_data == nullptr || output_data->empty()) {
      output_data = input_data->Clone();
      outputs.at(i) = output_data;
    } else {
      CHECK(output_data->shapes() == output_data->shapes());
      memcpy(output_data->data().memptr(), input_data->data().mem,
             sizeof(float) * input_data->size());
    }
    output_data->Reshape(shapes, true);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ViewLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& view_layer) {
  CHECK(op != nullptr) << "View operator is nullptr";
  const std::map<std::string, RuntimeParameter*>& params = op->params;
  if (params.find("shape") == params.end()) {
    LOG(ERROR) << "View layer missing shape";
    return ParseParameterAttrStatus::kParameterMissingShape;
  }

  const auto& shape =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("shape"));
  if (!shape) {
    LOG(ERROR) << "View layer missing shape";
    return ParseParameterAttrStatus::kParameterMissingShape;
  }
  view_layer = std::make_shared<ViewLayer>(shape->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

ViewLayer::ViewLayer(const std::vector<int32_t>& shapes) : Layer("view") {
  for (int shape : shapes) {
    shapes_.push_back(shape);
  }
}

LayerRegistererWrapper kViewGetInstance("Tensor.view", ViewLayer::GetInstance);

}  // namespace kuiper_infer
