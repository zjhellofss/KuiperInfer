//
// Created by fss on 22-12-9.
//
#include "flatten.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
FlattenLayer::FlattenLayer(int start_dim, int end_dim) : Layer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {

}

InferStatus FlattenLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                  std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

  int start_dim = start_dim_;
  int end_dim = end_dim_;
  int total_dims = 4; // NCHW

  if (start_dim < 0) {
    start_dim = total_dims + start_dim;
  }
  if (end_dim < 0) {
    end_dim = total_dims + end_dim;
  }

  end_dim -= 1;
  start_dim -= 1;
  CHECK(end_dim > start_dim);
  CHECK(end_dim <= 2 && start_dim >= 0);
  const uint32_t batch_size = inputs.size();
  if (outputs.empty()) {
    outputs = std::vector<std::shared_ptr<Tensor<float>>>(batch_size);
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    const auto &shapes = input->shapes();
    uint32_t elements_size = 1;

    for (int s = start_dim; s <= end_dim; ++s) {
      elements_size *= shapes.at(s);
    }
    std::shared_ptr<Tensor<float>> output = input->Clone();
    if (start_dim == 0 && end_dim == 2) {
      output->ReRawshape({elements_size});
    } else if (start_dim == 1 && end_dim == 2) {
      uint32_t channels = input->channels();
      output->ReRawshape({channels, elements_size});
    } else if (start_dim == 0 && end_dim == 1) {
      uint32_t cols = input->cols();
      output->ReRawshape({elements_size, cols});
    } else {
      LOG(FATAL) << "Wrong flatten dim: " << "start dim: " << start_dim << " end dim: " << end_dim;
    }
    if (outputs.size() > i) {
      outputs.at(i) = output;
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus FlattenLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                   std::shared_ptr<Layer> &flatten_layer) {
  const auto &params = op->params;
  if (params.find("end_dim") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  if (params.find("start_dim") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  const auto &start_dim = dynamic_cast<RuntimeParameterInt *>(params.at("start_dim"));
  const auto &end_dim = dynamic_cast<RuntimeParameterInt *>(params.at("end_dim"));
  if (start_dim == nullptr || end_dim == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  flatten_layer = std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kFlattenGetInstance("torch.flatten", FlattenLayer::GetInstance);

}