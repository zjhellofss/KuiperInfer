//
// Created by fss on 22-11-12.
//
#include "adaptive_avgpooling.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h,
                                                         uint32_t output_w)
    : Layer("AdaptiveAveragePooling"),
      output_h_(output_h),
      output_w_(output_w) {}

InferStatus AdaptiveAveragePoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR)
        << "The input tensor array in the adaptive pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the adaptive "
                  "pooling layer "
                  "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }


  


  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus AdaptiveAveragePoolingLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& avg_layer) {
  CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";

  const auto& output_hw =
      dynamic_cast<RuntimeParameterIntArray*>(params.at("output_size"));
  if (!output_hw) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  const auto& output_hw_arr = output_hw->value;
  if (output_hw_arr.size() != 2) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }
  avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(
      output_hw_arr.at(0), output_hw_arr.at(1));
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kAdaptiveAvgpoolingGetInstance(
    "nn.AdaptiveAvgPool2d", AdaptiveAveragePoolingLayer::GetInstance);

}  // namespace kuiper_infer