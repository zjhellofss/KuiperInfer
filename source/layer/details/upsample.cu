//
// Created by fss on 22-12-25.
//
#include <stdio.h>
#include "layer/abstract/layer_factory.hpp"
#include "upsample.hpp"
#include "utils/gpu_utils.cuh"
namespace kuiper_infer {

__global__ void UpSampleLayerForward_Nearest(const int n, const float* in,
                                             float* out, uint32_t nums,
                                             uint32_t channel, uint32_t height,
                                             uint32_t weight, float scale_h,
                                             float scale_w) {
  CUDA_KERNEL_LOOP(i, n) {

    uint32_t out_w =static_cast<uint32_t>(weight * scale_w);
    uint32_t out_h =static_cast<uint32_t>(height * scale_h);
   
    uint32_t w = i % out_w;
    uint32_t h = (i / out_w) %out_h;
    uint32_t c = (i / (out_h*out_w)) % channel;

    uint32_t n = i / (weight * scale_w * height * scale_h * channel);
    uint32_t in_w = static_cast<uint32_t>(w / scale_w);
    uint32_t in_h = static_cast<uint32_t>(h / scale_h);
    
    uint32_t in_index = n * channel * height * weight + c * height * weight +
                        in_h * weight + in_w;
    out[i] = in[in_index];
  }
}

UpSampleLayer::UpSampleLayer(float scale_h, float scale_w,
                             UpSampleMode mode)
    : Layer("upsample"), scale_h_(scale_h), scale_w_(scale_w), mode_(mode) {}

InferStatus UpSampleLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the upsample layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  LOG_IF(FATAL, this->mode_ != UpSampleMode::kModeNearest)
      << "Unsupported upsample mode: " << int(mode_);

  auto nums = inputs.at(0)->nums();
  auto weight = inputs.at(0)->cols();
  auto height = inputs.at(0)->rows();
  auto channels = inputs.at(0)->channels();

  auto test_scale_factor = [](int origin, float scale_factor) {
    float result = origin * scale_factor;
    if (std::abs(result - std::round(result)) > 0.0001) {
      LOG(ERROR) << "The input scale_factor is wrong";
    }
  };


  test_scale_factor(height, scale_h_);
  test_scale_factor(weight, scale_w_);

  if (outputs.empty()) {
    auto output = std::make_shared<Tensor<float>>(
        nums, channels, height * scale_h_, weight * scale_w_);
    outputs.push_back(output);
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR)
        << "The input and output tensor array size of the upsample layer do "
           "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  uint32_t count = nums * channels * height * scale_h_ * weight * scale_w_;

  UpSampleLayerForward_Nearest<<<KUIPER_GET_BLOCKS(count),
                                 KUIPER_CUDA_NUM_THREADS>>>(
      count, inputs.at(0)->gpu_data(), outputs.at(0)->gpu_data(), nums,
      channels, height, weight, scale_h_, scale_w_);

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus UpSampleLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& upsample_layer) {
  CHECK(op != nullptr) << "Upsample operator is null";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";
  if (params.find("scale_factor") == params.end()) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return ParseParameterAttrStatus::kParameterMissingScale;
  }

  const auto& scale_param = params.at("scale_factor");
  const auto& scales =
      dynamic_cast<RuntimeParameterFloatArray*>(params.at("scale_factor"));

  if (scales == nullptr) {
    LOG(ERROR) << "Can not find the scale factor parameter";
    return ParseParameterAttrStatus::kParameterMissingScale;
  }
  CHECK(scales->value.size() == 2) << "Scale factor need two dimension";

  if (params.find("mode") == params.end()) {
    LOG(ERROR) << "Can not find the mode parameter";
    return ParseParameterAttrStatus::kParameterMissingResizeMode;
  }

  const auto& mode = dynamic_cast<RuntimeParameterString*>(params.at("mode"));
  CHECK(mode->value == "nearest")
      << "The mode " << mode->value << " is not supported!";

  uint32_t scale_h = (uint32_t)(scales->value.at(0));
  uint32_t scale_w = (uint32_t)(scales->value.at(1));
  upsample_layer = std::make_shared<UpSampleLayer>(scale_h, scale_w);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kUpSamplerGetInstance("nn.Upsample",
                                             UpSampleLayer::GetInstance);
}  // namespace kuiper_infer