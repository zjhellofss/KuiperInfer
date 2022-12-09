//
// Created by fss on 22-11-12.
//
#include "view.hpp"
#include "runtime/runtime_ir.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

InferStatus ViewLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  CHECK(!shapes_.empty());
  CHECK(inputs.size() == outputs.size());

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    if (input_data->empty()) {
      LOG(ERROR) << "The input feature map of flatten layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    // 检查形状中-1的数量
    uint32_t total_size = input_data->size();
    uint32_t current_size = 1;
    int zero_index = -1;

    std::vector<uint32_t> shapes;
    CHECK(shapes_.front() != -1);
    for (int j = 1; j < shapes_.size(); ++j) {
      CHECK(shapes_.at(j) == -1 || shapes_.at(j) > 0);
      if (shapes_.at(j) == -1) {
        CHECK(zero_index == -1) << "Having two minus one in shape arrays";
        zero_index = j;
      } else {
        current_size *= shapes_.at(j);
        shapes.push_back(shapes_.at(j));
      }
    }

    CHECK(zero_index == -1 || zero_index == shapes_.size() - 1) << "Minus one shape is in the wrong axis";
    if (zero_index != -1) {
      CHECK(total_size >= current_size);
      shapes.push_back(uint32_t(total_size / current_size));
    }
    std::shared_ptr<Tensor<float>> output_data = input_data->Clone();
    output_data->ReRawshape(shapes);
    outputs.at(i) = output_data;
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ViewLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                std::shared_ptr<Layer> &view_layer) {

  const std::map<std::string, RuntimeParameter *> &params = op->params;
  if (params.find("shape") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingShape;
  }

  const auto &shape = dynamic_cast<RuntimeParameterIntArray *>(params.at("shape"));
  if (!shape) {
    return ParseParameterAttrStatus::kParameterMissingShape;
  }
  view_layer = std::make_shared<ViewLayer>(shape->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

ViewLayer::ViewLayer(const std::vector<int32_t> &shapes) : Layer("view") {
  for (int shape : shapes) {
    shapes_.push_back(shape);
  }
}

LayerRegistererWrapper kViewGetInstance("Tensor.view", ViewLayer::GetInstance);

}
