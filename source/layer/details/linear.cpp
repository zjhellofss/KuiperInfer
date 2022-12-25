//
// Created by fss on 22-11-13.
//

#include "linear.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

LinearLayer::LinearLayer(int32_t in_features, int32_t out_features, bool use_bias)
    : ParamLayer("Linear"), use_bias_(use_bias), in_features_(in_features), out_features_(out_features) {
  std::shared_ptr<Tensor<float>> weight = std::make_shared<Tensor<float>>(in_features, out_features, 1);
  this->weights_.push_back(weight);
  if (use_bias) {
    std::shared_ptr<Tensor<float>> bias = std::make_shared<Tensor<float>>(1, out_features, 1);
    bias->ReRawshape(std::vector<uint32_t>{(uint32_t) (out_features)});
    this->bias_.push_back(bias);
  }
}

InferStatus LinearLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                 std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of linear layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  if (this->weights_.empty()) {
    LOG(ERROR) << "The weight parameters is empty";
    return InferStatus::kInferFailedWeightParameterError;
  } else {
    if (this->use_bias_ && this->weights_.size() != this->bias_.size()) {
      return InferStatus::kInferFailedBiasParameterError;
      LOG(ERROR) << "The size of the weight and bias parameters is not equal";
    }
  }

  CHECK(weights_.size() == 1);
  if (use_bias_) {
    CHECK(bias_.size() == 1);
  }
  uint32_t batch = inputs.size();
  const std::shared_ptr<Tensor<float>> &weight = weights_.front();

#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    const std::vector<uint32_t> &raw_shapes = input->raw_shapes();
    CHECK(raw_shapes.size() == 2);
    const uint32_t feature_dims = raw_shapes.at(0);
    arma::fmat weight_data(weight->data().memptr(), in_features_, out_features_);
    CHECK(weight_data.n_cols == out_features_);
    CHECK(weight_data.n_rows == feature_dims && feature_dims == in_features_);

    arma::fmat col_vec(input->data().memptr(), 1, in_features_);
    arma::fmat result = (col_vec * weight_data).t();

    if (use_bias_) {
      CHECK(!this->bias_.empty());
      const auto &bias_cube = this->bias_.front();
      CHECK(!bias_cube->empty());

      const auto &bias_data = bias_cube->data();
      CHECK(bias_data.n_slices == 1);
      CHECK(bias_data.n_rows == out_features_);
      result += bias_data.slice(0);
    }

    auto &output = outputs.at(i);
    CHECK(output->channels() == 1);
    const auto &output_raw_shapes = output->raw_shapes();
    CHECK(output_raw_shapes.size() == 2);
    CHECK(output_raw_shapes.at(0) == out_features_ && output_raw_shapes.at(1) ==1);

    output->at(0) = std::move(result);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus LinearLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                  std::shared_ptr<Layer> &linear_layer) {

  const auto &params = op->params;
  if (params.find("bias") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  const auto &use_bias_param = dynamic_cast<RuntimeParameterBool *>(params.at("bias"));
  if (use_bias_param == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  const auto &attr = op->attribute;
  if (attr.find("weight") == attr.end()) {
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  if (use_bias_param->value) {
    if (attr.find("bias") == attr.end()) {
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
  }

  const auto &weight = attr.at("weight");
  const auto &bias = attr.at("bias");
  const auto &shapes = weight->shape;
  CHECK(shapes.size() == 2) << "The graph only support two dimension matrix multiply";

  int32_t out_features = shapes.at(0);
  int32_t in_features = shapes.at(1);
  const bool use_bias = use_bias_param->value;

  linear_layer = std::make_shared<LinearLayer>(in_features, out_features, use_bias);
  if (use_bias) {
    linear_layer->set_bias(bias->get<float>());
  }

  // load weights
  linear_layer->set_weights(weight->get<float>());
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kLinearGetInstance("nn.Linear", LinearLayer::GetInstance);

}