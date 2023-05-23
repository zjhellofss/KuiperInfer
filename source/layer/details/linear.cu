//
//
//

#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "linear.hpp"

namespace kuiper_infer {

/**
 * w h代表最终生成的矩阵的宽和高
 *
 */
__global__ void operator_matmul(const float* input, const float* weight,
                                float* output, int w, int k, int h) {
  float sum = 0;

  float* input_base =
      input + blockIdx.x * blockDim.y * w * k + blockIdx.y * w * k;

  int row = threadIdx.y;
  int col = threadIdx.x;

  for (int ki = 0; ki < k; ++ki) {
    sum += input_base[row * k + ki] * weight[ki * h + col];
  }

  int loc =
      blockIdx.x * blockDim.y * w * h + blockIdx.y * w * h + row * h + col;
  
  output[loc] = sum;
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

  if (this->weights_.size() != 1) {
    LOG(ERROR) << "Need one weight tensor in the linear layer";
    return InferStatus::kInferFailedWeightParameterError;
  }

  if (use_bias_ && this->bias_.size() != 1) {
    LOG(ERROR) << "Need one bias tensor in the linear layer";
    return InferStatus::kInferFailedBiasParameterError;
  }

  float* weight_data = this->weights_.at(0).get()->gpu_data();

  uint32_t batch = inputs.at(0).get()->nums();
  uint32_t channels = inputs.at(0).get()->channels();
  uint32_t w = inputs.at(0).get()->rows();
  uint32_t k = inputs.at(0).get()->cols();
  uint32_t h = this->weights_.at(0).get()->cols();

  const std::shared_ptr<Tensor<float>>& weight = weights_.front();

  CHECKEQ(inputs.at(0).get()->cols(), this->weights_.at(0).get()->rows());

  if (output == nullptr || output->empty()) {
    output = std::make_shared<Tensor<float>>(1, out_features_, feature_dims);
    outputs.at(i) = output;
  }

  dim3 gridSize(batch, channels);
  dim3 blockSize(rows, cols);

  operator_matmul<<<gridSize, blockSize>>>(inputs_data, weight_data, );

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

  int32_t in_features = shapes.at(0);
  int32_t out_features = shapes.at(1);

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