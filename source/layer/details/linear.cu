//
//
//

#include <glog/logging.h>
#include <stdio.h>
#include <iostream>
#include "layer/abstract/layer_factory.hpp"
#include "linear.hpp"
#include "utils/gpu_utils.cuh"

namespace kuiper_infer {

/**
 * w h代表最终生成的矩阵的宽和高
 *
 */
__global__ void LinearForward(int n, float* output, const float* input,
                              const float* weight, int channels, int w, int k,
                              int h) {
  // weight h * k
  // input  batch *channel w * k
  // output batch *channel* w * h
  CUDA_KERNEL_LOOP(index, n) {
    int batch = index / (channels * w * h);
    int channel = (index / (w * h)) % channels;
    int row = (index / h) % w;
    int col = index % h;
    const float* input_base =
        input + batch * channels * w * k + channel * w * k + row * k;

    const float* wegiht_base = weight + col * h;
    float sum = 0.0;
    // weight 采用列存储的方式
    for (int ki = 0; ki < k; ++ki) {
      sum += input_base[ki] * weight[ki];
    }
    int loc = batch * channels * w * h + channel * w * h + row * h + col;
    output[loc] = sum;
  }
}

__global__ void add_bias(int n, float* output, const float* input,
                         const float* bias, int w, int h) {
  CUDA_KERNEL_LOOP(index, n) {
    int col = index % h;

    output[index] = input[index] + bias[col];
  }
}

LinearLayer::LinearLayer(int32_t in_features, int32_t out_features,
                         bool use_bias)
    : ParamLayer("Linear"),
      use_bias_(use_bias),
      in_features_(in_features),
      out_features_(out_features) {
  this->InitWeightParam(1, 1, out_features, in_features);
  if (use_bias) {
    this->InitBiasParam(1, 1, 1, out_features);
  }
}

InferStatus LinearLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the linear layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of linear layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (this->weights_.empty()) {
    LOG(ERROR) << "The weight tensor in the linear layer is empty";
    return InferStatus::kInferFailedWeightParameterError;
  } else {
    if (this->use_bias_ && this->weights_.size() != this->bias_.size()) {
      LOG(ERROR) << "The size of the weight and bias tensor do not match";
      return InferStatus::kInferFailedBiasParameterError;
    }
  }
  uint32_t batch = inputs.at(0).get()->nums();
  uint32_t channels = inputs.at(0).get()->channels();
  uint32_t w = inputs.at(0).get()->rows();
  uint32_t k = inputs.at(0).get()->cols();

  CHECK(k == in_features_);
  uint32_t h = this->weights_.at(0).get()->rows();
  CHECK(h == out_features_);

  if (outputs.empty()) {
    auto output =
        std::make_shared<Tensor<float>>(batch, channels, w, out_features_);
    outputs.push_back(output);
  }

  float* weight_data = this->weights_.at(0).get()->gpu_data();
  CHECK(weight_data != nullptr);

  dim3 gridSize(batch, channels);
  dim3 blockSize(w, h);

  float* top_data = outputs.at(0)->gpu_data();
  float* bottom_data = inputs.at(0)->gpu_data();

  auto count = batch * channels * w * h;

  LinearForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, top_data, bottom_data, weight_data, channels, w, k, h);
  if (use_bias_) {
    float* bias_data = this->bias_.at(0)->gpu_data();
    float* bottom_data = outputs.at(0).get()->gpu_data();
    int count = outputs.at(0)->size();

    add_bias<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_data, bias_data, w, h);
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