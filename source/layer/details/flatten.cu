//
// Created by fss on 22-12-9.
//
#include "data/tensor_util.hpp"
#include "flatten.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
FlattenLayer::FlattenLayer(int start_dim, int end_dim)
    : Layer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

InferStatus FlattenLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the flatten "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  int start_dim = start_dim_;
  int end_dim = end_dim_;
  int total_dims = 4;  // NCHW

  if (start_dim < 0) {
    start_dim = total_dims + start_dim;
  }
  if (end_dim < 0) {
    end_dim = total_dims + end_dim;
  }

  CHECK(end_dim > start_dim) << "The end dim must greater than start dim";
  CHECK(end_dim <= 3 && start_dim >= 1)
      << "The end dim must less than two and start dim must greater than zero";

  if (outputs.empty()) {
    std::shared_ptr<Tensor<float>> output =
        std::make_shared<Tensor<float>>(inputs[0]->shapes());
    outputs.push_back(output);
  }

  std::vector<uint32_t> output_shape;
  auto inputs_shape = inputs[0]->shapes();
  uint32_t flatten_dim = 1;

  for (int i = 0; i < inputs_shape.size(); i++) {
    if (i >= start_dim && i <= end_dim) {
      flatten_dim *= inputs_shape[i];
      if (i == end_dim) {
        output_shape.push_back(flatten_dim);
      }
    } else {
      output_shape.push_back(inputs_shape[i]);
    }
  }
  
  *outputs.at(0).get() = *inputs.at(0).get();
  outputs[0]->Reshape(output_shape);

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus FlattenLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& flatten_layer) {
  CHECK(op != nullptr) << "Flatten operator is nullptr";
  const auto& params = op->params;

  if (params.find("end_dim") == params.end()) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  if (params.find("start_dim") == params.end()) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  const auto& start_dim =
      dynamic_cast<RuntimeParameterInt*>(params.at("start_dim"));

  const auto& end_dim =
      dynamic_cast<RuntimeParameterInt*>(params.at("end_dim"));

  if (start_dim == nullptr || end_dim == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  flatten_layer =
      std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kFlattenGetInstance("torch.flatten",
                                           FlattenLayer::GetInstance);

}  // namespace kuiper_infer