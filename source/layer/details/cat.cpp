//
// Created by fss on 22-12-25.
//
#include "cat.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
CatLayer::CatLayer(int dim) : Layer("cat"), dim_(dim) {

}

InferStatus CatLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() == outputs.size()) {
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }
  if (dim_ != 1 && dim_ != -3) {
    return InferStatus::kInferFailedDimensionParameterError;
  }

  const uint32_t batch_size = outputs.size();
  CHECK(inputs.size() % outputs.size() == 0);
  const uint32_t packet_size = inputs.size() / outputs.size();

  for (uint32_t i = 0; i < batch_size; ++i) {
    const uint32_t pos_index = i * packet_size;
    uint32_t total_channels = 0;
    const uint32_t rows = inputs.at(pos_index)->rows();
    const uint32_t cols = inputs.at(pos_index)->cols();

    for (uint32_t j = 0; j < packet_size; ++j) {
      total_channels += inputs.at(j + pos_index)->channels();
      CHECK(rows == inputs.at(j + pos_index)->rows());
      CHECK(cols == inputs.at(j + pos_index)->cols());
    }

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(total_channels, rows, cols);
    }

    CHECK(output->channels() == total_channels && output->rows() == rows && output->cols() == cols);
    for (uint32_t j = 0; j < packet_size; ++j) {
      const std::shared_ptr<Tensor<float>> &input = inputs.at(j + pos_index);
      const uint32_t channels = input->channels();
      for (uint32_t c = 0; c < channels; ++c) {
        output->at(j * channels + c) = input->at(c);
      }
    }
    outputs.at(i) = output;
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus CatLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                               std::shared_ptr<Layer> &cat_layer) {
  CHECK(op != nullptr) << "Cat operator is nullptr";
  const auto &params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";
  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  const auto &dim_param = dynamic_cast<RuntimeParameterInt *>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kCatGetInstance("torch.cat", CatLayer::GetInstance);

}