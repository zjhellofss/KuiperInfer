//
// Created by fss on 22-12-12.
//
#include "hardsigmoid.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

__global__ void HardSigmoidForward(const int n, const float* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = in[index];
    float result = 0.f;
    if (val <= -3.f) {
      result = 0.f;
    } else if (val >= 3.f) {
      result = 1.f;
    } else {
      result = val / 6.f + 0.5f;
    }
    out[index] = result;
  }
}

HardSigmoid::HardSigmoid() : Layer("HardSigmoid") {}

InferStatus HardSigmoid::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the hardsigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the hardsigmoid "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (outputs.empty()) {
    std::shared_ptr<Tensor<float>> output =
        std::make_shared<Tensor<float>>(inputs.at(0)->shapes());
    outputs.push_back(output);
  } else {
    CHECK(inputs.at(0)->shapes() == outputs.at(0)->shapes())
        << "The input tensor and output tensor in the relu layer are not the "
           "same shape";
  }

  uint32_t count = inputs.at(0)->size();
  HardSigmoidForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, inputs.at(0)->gpu_data(), outputs.at(0)->gpu_data());

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus HardSigmoid::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& hardsigmoid_layer) {
  CHECK(op != nullptr) << "HardSigmoid operator is nullptr";
  hardsigmoid_layer = std::make_shared<HardSigmoid>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kHardSigmoidGetInstance("nn.Hardsigmoid",
                                               HardSigmoid::GetInstance);

}  // namespace kuiper_infer
