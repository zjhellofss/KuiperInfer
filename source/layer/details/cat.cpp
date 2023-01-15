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
    LOG(ERROR) << "The input feature map of cat layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() == outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension of cat layer is error";
    return InferStatus::kInferFailedDimensionParameterError;
  }

  const uint32_t output_size = outputs.size();
  CHECK(inputs.size() % output_size == 0);
  const uint32_t packet_size = inputs.size() / output_size;
//#pragma omp parallel for num_threads(outputs.size())
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    uint32_t start_channel = 0;
    uint32_t rows = inputs.front()->rows();
    uint32_t cols = inputs.front()->cols();

    for (uint32_t j = i; j < inputs.size(); j += output_size) {
      const std::shared_ptr<Tensor<float>> &input = inputs.at(j);
      const uint32_t in_channels = input->channels();
      CHECK(rows == input->rows() && cols == input->cols());

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(in_channels * packet_size, rows, cols);
        outputs.at(i) = output;
      }
      CHECK(output->channels() == in_channels * packet_size && output->rows() == rows && output->cols() == cols);
      for (uint32_t c = 0; c < in_channels; ++c) {
        output->at(start_channel + c) = input->at(c);
      }
      start_channel += input->channels();
    }
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