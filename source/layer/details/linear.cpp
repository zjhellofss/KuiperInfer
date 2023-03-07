//
// Created by fss on 22-11-13.
//

#include "linear.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

LinearLayer::LinearLayer(int32_t in_features, int32_t out_features,
                         bool use_bias)
    : ParamLayer("Linear"),
      use_bias_(use_bias),
      in_features_(in_features),
      out_features_(out_features) {
  this->InitWeightParam(1, 1, out_features, in_features);
  if (use_bias) {
    this->InitBiasParam(1, 1, out_features, 1);
  }
  // std::shared_ptr<Tensor<float>> weight =
  //     std::make_shared<Tensor<float>>(1, out_features, in_features);
  // this->weights_.push_back(weight);
  // if (use_bias) {
  //   std::shared_ptr<Tensor<float>> bias =
  //       std::make_shared<Tensor<float>>(1, out_features, 1);
  //   bias->ReRawshape(std::vector<uint32_t>{(uint32_t)(out_features)});
  //   this->bias_.push_back(bias);
  // }
}

InferStatus LinearLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
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

  if (weights_.size() != 1) {
    LOG(ERROR) << "The size of weight parameters is not one";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (use_bias_ && this->bias_.size() != 1) {
    LOG(ERROR) << "The size of bias parameters is not one";
    return InferStatus::kInferFailedBiasParameterError;
  }
  uint32_t batch = inputs.size();
  const std::shared_ptr<Tensor<float>>& weight = weights_.front();
  arma::fmat weight_data(weight->data().memptr(), out_features_, in_features_,
                         false, true);

#pragma omp parallel for num_threads(batch)
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input feature map of linear layer is empty";
    const std::vector<uint32_t>& input_shapes = input->shapes();
    CHECK(input_shapes.size() == 3 && input_shapes.front() == 1);

    const uint32_t feature_dims = input_shapes.at(1);
    CHECK(weight_data.n_rows == out_features_);
    CHECK(weight_data.n_cols == feature_dims && feature_dims == in_features_);
    const uint32_t input_dim = input_shapes.at(2);

    arma::fmat col_vec(input->data().memptr(), in_features_, input_dim, false,
                       true);

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(1, out_features_, input_dim);
      outputs.at(i) = output;
    }
    CHECK(output->channels() == 1 && output->rows() == out_features_ &&
          output->cols() == input_dim);
    const auto& output_raw_shapes = output->raw_shapes();
    CHECK(output_raw_shapes.size() == 2);
    CHECK(output_raw_shapes.at(0) == out_features_ &&
          output_raw_shapes.at(1) == input_dim);

    arma::fmat& result = output->slice(0);
    result = weight_data * col_vec;
    if (use_bias_) {
      CHECK(!this->bias_.empty() && this->bias_.size() == 1);
      const auto& bias_cube = this->bias_.front();
      CHECK(!bias_cube->empty());

      const auto& bias_data = bias_cube->data();
      CHECK(bias_data.n_slices == 1);
      CHECK(bias_data.n_rows == out_features_);
      result += bias_data.slice(0);
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus LinearLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& linear_layer) {
  CHECK(op != nullptr) << "Linear operator is nullptr";
  const auto& params = op->params;
  if (params.find("bias") == params.end()) {
    LOG(ERROR) << "Can not find the use bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }
  const auto& use_bias_param =
      dynamic_cast<RuntimeParameterBool*>(params.at("bias"));
  if (use_bias_param == nullptr) {
    LOG(ERROR) << "Can not find the use bias parameter";
    return ParseParameterAttrStatus::kParameterMissingUseBias;
  }

  const auto& attr = op->attribute;
  CHECK(!attr.empty()) << "Operator attributes is empty";

  if (attr.find("weight") == attr.end()) {
    LOG(ERROR) << "Can not find the weight parameter";
    return ParseParameterAttrStatus::kAttrMissingWeight;
  }

  if (use_bias_param->value) {
    if (attr.find("bias") == attr.end()) {
      LOG(ERROR) << "Can not find the bias parameter";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
  }

  const auto& weight = attr.at("weight");
  const auto& bias = attr.at("bias");
  const auto& shapes = weight->shape;
  if ((shapes.size() < 2)) {
    LOG(ERROR) << "The graph only support two dimension matrix multiply";
    return ParseParameterAttrStatus::kAttrMissingOutFeatures;
  }

  int32_t out_features = shapes.at(0);
  int32_t in_features = shapes.at(1);
  const bool use_bias = use_bias_param->value;

  linear_layer =
      std::make_shared<LinearLayer>(in_features, out_features, use_bias);
  if (use_bias) {
    linear_layer->set_bias(bias->get<float>());
  }

  // load weights
  linear_layer->set_weights(weight->get<float>());
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kLinearGetInstance("nn.Linear",
                                          LinearLayer::GetInstance);

}  // namespace kuiper_infer