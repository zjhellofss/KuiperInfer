//
// Created by fss on 22-11-17.
//
#include "batchnorm2d.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "parser/runtime_ir.hpp"

namespace kuiper_infer {

InferStatus BatchNorm2dLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                      std::vector<std::shared_ptr<Tensor>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of batchnorm layer is empty";
    return InferStatus::kInferFailedInputEmpty;
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
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of batchnorm layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }

    std::shared_ptr<Tensor> output = input->Clone();
    for (uint32_t i = 0; i < mean_value_size; ++i) {
      const double mean_value = weights_.at(i)->index(0);
      const double var_value = bias_.at(i)->index(0);

      if (input->channels() < i) {
        LOG(ERROR) << "The channel of the input feature maps and mean values is not adapting";
        return InferStatus::kInferFailedChannelParameterError;
      }
      const uint32_t input_row = input->rows();
      const uint32_t input_col = input->cols();

      auto var_mat = arma::mat(input_row, input_col);
      var_mat.fill(var_value + eps_);
      var_mat = arma::sqrt(var_mat);

      auto mean_mat = arma::mat(input_row, input_col);
      mean_mat.fill(mean_value);

      output->at(i) = (output->at(i) - mean_mat) / var_mat;
      outputs.push_back(output);
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus BatchNorm2dLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                       std::shared_ptr<Layer> &batch_layer) {

  const auto &params = op->params;
  if (params.find("eps") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingEps;
  }
  const auto &eps_p = dynamic_cast<RuntimeParameterFloat *>(params.at("eps"));
  const float eps_value = eps_p->value;

  if (params.find("num_features") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingNumFeatures;
  }

  const auto &num_features_p = dynamic_cast<RuntimeParameterInt *>(params.at("num_features"));
  const int num_features = num_features_p->value;
  batch_layer = std::make_shared<BatchNorm2dLayer>(num_features, eps_value);

  // load weights
  const auto &attrs = op->attribute;
  const auto &mean_attr_iter = attrs.find("running_mean");
  if (mean_attr_iter == attrs.end()) {
    return ParseParameterAttrStatus::kAttrMissingRunningMean;
  }

  const auto &mean_attr = mean_attr_iter->second;
  std::vector<double> mean = mean_attr->get<double>();
  batch_layer->set_weights(mean);

  const auto &var_attr_iter = attrs.find("running_var");
  if (var_attr_iter == attrs.end()) {
    return ParseParameterAttrStatus::kAttrMissingRunningVar;
  }

  const auto &var_attr = var_attr_iter->second;
  std::vector<double> var = var_attr->get<double>();;
  batch_layer->set_bias(var);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

BatchNorm2dLayer::BatchNorm2dLayer(uint32_t num_features, double eps)
    : ParamLayer("Batchnorm"), num_features_(num_features), eps_(eps) {
  for (uint32_t i = 0; i < num_features; ++i) {
    std::shared_ptr<Tensor> weight = std::make_shared<Tensor>(1, 1, 1);
    this->weights_.push_back(weight);

    std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(1, 1, 1);
    this->bias_.push_back(bias);
  }
}

LayerRegistererWrapper kBatchNorm2dGetInstance("nn.BatchNorm2d", BatchNorm2dLayer::GetInstance);

}