//
// Created by fss on 22-12-9.
//
#include "flatten.hpp"

#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
FlattenLayer::FlattenLayer(int start_dim, int end_dim)
    : Layer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

InferStatus FlattenLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
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

  end_dim -= 1;
  start_dim -= 1;
  CHECK(end_dim > start_dim) << "End dim must greater than start dim";
  CHECK(end_dim <= 2 && start_dim >= 0)
          << "end dim must less than two and start dim must greater than zero";

  const uint32_t batch_size = inputs.size();

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input feature map of flatten layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }

    const auto &shapes = input->shapes();
    uint32_t elements_size = 1;

    for (int s = start_dim; s <= end_dim; ++s) {
      elements_size *= shapes.at(s);
    }

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = input->Clone();
      outputs.at(i) = output;
    } else {
      CHECK(outputs.at(i)->shapes() == output->shapes());
      memcpy(output->data().memptr(), input->data().memptr(), sizeof(float) * input->size());
    }
    if (start_dim == 0 && end_dim == 2) {
      output->Reshape({elements_size}, true);
    } else if (start_dim == 1 && end_dim == 2) {
      uint32_t channels = input->channels();
      output->Reshape({channels, elements_size}, true);
    } else if (start_dim == 0 && end_dim == 1) {
      uint32_t cols = input->cols();
      output->Reshape({elements_size, cols}, true);
    } else {
      LOG(FATAL) << "Wrong flatten dim: "
                 << "start dim: " << start_dim << " end dim: " << end_dim;
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus FlattenLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &flatten_layer) {
  CHECK(op != nullptr) << "Flatten operator is nullptr";
  const auto &params = op->params;

  if (params.find("end_dim") == params.end()) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  if (params.find("start_dim") == params.end()) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  const auto &start_dim =
      dynamic_cast<RuntimeParameterInt *>(params.at("start_dim"));

  const auto &end_dim =
      dynamic_cast<RuntimeParameterInt *>(params.at("end_dim"));

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