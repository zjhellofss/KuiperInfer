//
// Created by fss on 22-11-17.
//
#include "batchnorm2d.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"

namespace kuiper_infer {

InferStatus BatchNorm2dLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
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
    LOG(ERROR) << "BatchNorm2d layer do not have the same mean values and bias values";
    return InferStatus::kInferFailedWeightParameterError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t b = 0; b < batch_size; ++b) {
    const auto &input = inputs.at(b);
    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input feature map of batchnorm layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }

    if (input->channels() != mean_value_size) {
      LOG(ERROR) << "The channel of of input and mean value mat is not equal";
      return InferStatus::kInferFailedChannelParameterError;
    }

    std::shared_ptr<Tensor<float>> output = outputs.at(b);
    if (output == nullptr || output->empty()) {
      LOG(ERROR) << "The output size of batchnorm is empty";
      output = std::make_shared<Tensor<float>>(input->channels(), input->rows(), input->cols());
    }

    CHECK(output->shapes() == input->shapes()) << "The output size of batchnorm is error";

    for (uint32_t i = 0; i < mean_value_size; ++i) {
      const float mean_value = weights_.at(i)->index(0);
      const float var_value = bias_.at(i)->index(0);

      if (input->channels() < i) {
        LOG(ERROR) << "The channel of the input feature maps and mean values is not adapting";
        return InferStatus::kInferFailedChannelParameterError;
      }

      float var_value_ = var_value + eps_;
      var_value_ = std::sqrt(var_value_);
      output->at(i) = (input->at(i) - mean_value) / var_value_;
    }
    outputs.at(b) = output;
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus BatchNorm2dLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &batch_layer) {
  CHECK(op != nullptr) << "BatchNorm get instance failed, operator is nullptr";

  const auto &params = op->params;
  if (params.find("eps") == params.end()) {
    LOG(ERROR) << "Can not find the eps parameter";
    return ParseParameterAttrStatus::kParameterMissingEps;
  }

  const auto &eps = dynamic_cast<RuntimeParameterFloat *>(params.at("eps"));
  if (!eps) {
    LOG(ERROR) << "Can not find the eps parameter";
    return ParseParameterAttrStatus::kParameterMissingEps;
  }

  if (params.find("num_features") == params.end()) {
    LOG(ERROR) << "Can not find the num features parameter";
    return ParseParameterAttrStatus::kParameterMissingNumFeatures;
  }

  const auto &num_features = dynamic_cast<RuntimeParameterInt *>(params.at("num_features"));
  if (!num_features) {
    LOG(ERROR) << "Can not find the num features parameter";
    return ParseParameterAttrStatus::kParameterMissingNumFeatures;
  }

  batch_layer = std::make_shared<BatchNorm2dLayer>(num_features->value, eps->value);

  // load weights
  const auto &attrs = op->attribute;
  if (attrs.find("running_mean") == attrs.end()) {
    LOG(ERROR) << "Can not find the running mean attribute";
    return ParseParameterAttrStatus::kAttrMissingRunningMean;
  }

  const auto &mean_attr = attrs.at("running_mean");
  std::vector<float> mean = mean_attr->get<float>();
  batch_layer->set_weights(mean);

  const auto &var_attr_iter = attrs.find("running_var");
  if (var_attr_iter == attrs.end()) {
    LOG(ERROR) << "Can not find the running var attribute";
    return ParseParameterAttrStatus::kAttrMissingRunningVar;
  }

  const auto &var_attr = var_attr_iter->second;
  std::vector<float> var = var_attr->get<float>();;
  batch_layer->set_bias(var);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

BatchNorm2dLayer::BatchNorm2dLayer(uint32_t num_features, float eps)
    : ParamLayer("Batchnorm"), num_features_(num_features), eps_(eps) {
  for (uint32_t i = 0; i < num_features; ++i) {
    std::shared_ptr<Tensor<float>> weight = std::make_shared<Tensor<float>>(1, 1, 1);
    this->weights_.push_back(weight);

    std::shared_ptr<Tensor<float>> bias = std::make_shared<Tensor<float>>(1, 1, 1);
    this->bias_.push_back(bias);
  }
}

LayerRegistererWrapper kBatchNorm2dGetInstance("nn.BatchNorm2d", BatchNorm2dLayer::GetInstance);

}