//
// Created by fss on 22-11-17.
//
#include "batchnorm2d.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

InferStatus BatchNorm2dLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of batchnorm layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t mean_value_size = this->weights_.size();
  const uint32_t bias_value_size = this->bias_.size();
  if (mean_value_size != bias_value_size) {
    LOG(ERROR)
        << "BatchNorm2d layer do not have the same mean values and bias values";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (this->affine_bias_.size() != this->affine_weight_.size()) {
    LOG(ERROR) << "BatchNorm2d layer do not have the same affine weight and "
                  "bias values";
    return InferStatus::kInferFailedWeightParameterError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto& input_data = inputs.at(i);
    const auto& output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input feature map of batchNorm2d layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
      }
    }
  }

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto& input = inputs.at(b);
    CHECK(input != nullptr && !input->empty())
        << "The input feature map of batchnorm layer is empty";
    CHECK(input->channels() == mean_value_size)
        << "The channel of of input and mean value mat is not equal";
    CHECK(input->channels() == affine_weight_.size())
        << "The channel of input and affine weight is not equal";

    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The output size of batchnorm is empty";
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(b) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The output size of batchnorm is error";

    for (uint32_t i = 0; i < mean_value_size; ++i) {
      CHECK(weights_.at(i)->size() == 1 && bias_.at(i)->size() == 1);
      const float mean_value = weights_.at(i)->index(0);
      const float var_value = bias_.at(i)->index(0);
      CHECK(input->channels() >= i) << "The channel of the input feature maps "
                                       "and mean values is not adapting";

      const float var_value_ = std::sqrt(var_value + eps_);
      output->at(i) =
          ((input->at(i) - mean_value) / var_value_) * affine_weight_.at(i) +
          affine_bias_.at(i);
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus BatchNorm2dLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& batch_layer) {
  CHECK(op != nullptr) << "BatchNorm get instance failed, operator is nullptr";

  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";

  if (params.find("eps") == params.end()) {
    LOG(ERROR) << "Can not find the eps parameter";
    return ParseParameterAttrStatus::kParameterMissingEps;
  }

  const auto& eps = dynamic_cast<RuntimeParameterFloat*>(params.at("eps"));
  if (!eps) {
    LOG(ERROR) << "Can not find the eps parameter";
    return ParseParameterAttrStatus::kParameterMissingEps;
  }

  if (params.find("num_features") == params.end()) {
    LOG(ERROR) << "Can not find the num features parameter";
    return ParseParameterAttrStatus::kParameterMissingNumFeatures;
  }

  const auto& num_features =
      dynamic_cast<RuntimeParameterInt*>(params.at("num_features"));
  if (!num_features) {
    LOG(ERROR) << "Can not find the num features parameter";
    return ParseParameterAttrStatus::kParameterMissingNumFeatures;
  }

  // load weights
  const auto& attrs = op->attribute;
  CHECK(!attrs.empty()) << "Operator attributes is empty";

  if (attrs.find("running_mean") == attrs.end()) {
    LOG(ERROR) << "Can not find the running mean attribute";
    return ParseParameterAttrStatus::kAttrMissingRunningMean;
  }

  if (attrs.find("weight") == attrs.end()) {
    LOG(ERROR) << "Can not find the affine weight attribute";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }
  if (attrs.find("bias") == attrs.end()) {
    LOG(ERROR) << "Can not find the affine bias attribute";
    return ParseParameterAttrStatus::kAttrMissingBias;
  }

  const std::vector<float>& affine_weight = attrs.at("weight")->get<float>();
  const std::vector<float>& affine_bias = attrs.at("bias")->get<float>();
  batch_layer = std::make_shared<BatchNorm2dLayer>(
      num_features->value, eps->value, affine_weight, affine_bias);

  const auto& mean_attr = attrs.at("running_mean");
  const std::vector<float>& mean = mean_attr->get<float>();
  batch_layer->set_weights(mean);

  if (attrs.find("running_var") == attrs.end()) {
    LOG(ERROR) << "Can not find the running var attribute";
    return ParseParameterAttrStatus::kAttrMissingRunningVar;
  }

  const auto& var_attr = attrs.at("running_var");
  const std::vector<float>& var = var_attr->get<float>();
  batch_layer->set_bias(var);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

BatchNorm2dLayer::BatchNorm2dLayer(uint32_t num_features, float eps,
                                   const std::vector<float>& affine_weight,
                                   const std::vector<float>& affine_bias)
    : ParamLayer("Batchnorm"),
      affine_weight_(affine_weight),
      affine_bias_(affine_bias),
      eps_(eps) {
  this->InitWeightParam(num_features, 1, 1, 1);
  this->InitBiasParam(num_features, 1, 1, 1);
  // for (uint32_t i = 0; i < num_features; ++i) {
  //   std::shared_ptr<Tensor<float>> weight =
  //   std::make_shared<Tensor<float>>(1, 1, 1);
  //   this->weights_.push_back(weight);

  //   std::shared_ptr<Tensor<float>> bias = std::make_shared<Tensor<float>>(1,
  //   1, 1); this->bias_.push_back(bias);
  // }
}

LayerRegistererWrapper kBatchNorm2dGetInstance("nn.BatchNorm2d",
                                               BatchNorm2dLayer::GetInstance);

}  // namespace kuiper_infer