//
// Created by fss on 22-12-25.
//
#include "cat.hpp"
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
CatLayer::CatLayer(int dim) : NonParamLayer("cat"), dim_(dim) {}

InferStatus CatLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the cat layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() == outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the cat layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (dim_ != 1 && dim_ != -3) {
    LOG(ERROR) << "The dimension parameter of cat layer is error";
    return InferStatus::kInferFailedDimensionParameterError;
  }

  const uint32_t output_size = outputs.size();
  if (inputs.size() % output_size != 0) {
    LOG(ERROR)
        << "The input and output tensor array size of cat layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t packet_size = inputs.size() / output_size;
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    const std::shared_ptr<ftensor>& input_data = inputs.at(i);
    const std::shared_ptr<ftensor>& output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the cat layer has an "
                    "empty tensor "
                 << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data == nullptr || output_data->empty()) {
      continue;
    }
    const uint32_t input_channel = input_data->channels();
    const uint32_t output_channel = output_data->channels();
    if (input_channel * packet_size != output_channel) {
      LOG(ERROR) << "Inconsistent number of channels between input tensor and "
                    "output tensor";
      return InferStatus::kInferFailedChannelParameterError;
    }
    if (input_data->rows() != output_data->rows() ||
        input_data->cols() != output_data->cols()) {
      LOG(ERROR) << "The output and input shapes of the cat layer do "
                    "not match, "
                 << i << "th";
      return InferStatus::kInferFailedInputOutSizeMatchError;
    }
  }
#pragma omp parallel for num_threads(outputs.size())
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    uint32_t start_channel = 0;
    uint32_t rows = inputs.front()->rows();
    uint32_t cols = inputs.front()->cols();

    for (uint32_t j = i; j < inputs.size(); j += output_size) {
      const std::shared_ptr<Tensor<float>>& input = inputs.at(j);
      CHECK(input != nullptr && !input->empty())
          << "The input tensor array in the cat layer has "
             "an empty tensor "
          << j << " th";
      const uint32_t in_channels = input->channels();
      CHECK(rows == input->rows() && cols == input->cols())
          << "The input tensor array in the cat layer "
             "has an incorrectly sized tensor "
          << j << " th";

      if (output == nullptr || output->empty()) {
        output = std::make_shared<Tensor<float>>(in_channels * packet_size,
                                                 rows, cols);
        outputs.at(i) = output;
      }
      CHECK(output->channels() == in_channels * packet_size &&
            output->rows() == rows && output->cols() == cols)
          << "The output tensor array in the cat layer "
             "has an incorrectly sized tensor "
          << i << " th";
      const uint32_t plane_size = rows * cols;
      memcpy(output->raw_ptr(start_channel * plane_size), input->raw_ptr(),
             sizeof(float) * plane_size * in_channels);
      start_channel += input->channels();
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus CatLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& cat_layer) {
  CHECK(op != nullptr) << "Cat operator is nullptr";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";
  if (params.find("dim") == params.end()) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  const auto& dim_param = dynamic_cast<RuntimeParameterInt*>(params.at("dim"));
  if (!dim_param) {
    LOG(ERROR) << "Can not find the dim parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  const int32_t dim = dim_param->value;
  cat_layer = std::make_shared<CatLayer>(dim);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kCatGetInstance("torch.cat", CatLayer::GetInstance);
}  // namespace kuiper_infer