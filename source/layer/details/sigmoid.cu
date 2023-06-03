//
// Created by fss on 22-11-13.
//

#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#include "sigmoid.hpp"

namespace kuiper_infer {

__global__ void SigmoidForward(const int n, const float* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = in[index];
    out[index] = 1.0 / (1.f + expf(-val));
  }
}

InferStatus SigmoidLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the sigmoid layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the sigmoid layer "
                  "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  if (outputs.empty()) {
    std::shared_ptr<Tensor<float>> output =
        std::make_shared<Tensor<float>>(inputs[0]->shapes());
    outputs.push_back(output);
  } else {
    CHECK(inputs.at(0)->shapes() == outputs.at(0)->shapes())
        << "The input tensor and output tensor in the relu layer are not the "
           "same shape";
  }
  int count = inputs.at(0)->size();

  SigmoidForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, inputs.at(0)->gpu_data(), outputs.at(0)->gpu_data());

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus SigmoidLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& sigmoid_layer) {
  CHECK(op != nullptr) << "Sigmoid operator is nullptr";
  sigmoid_layer = std::make_shared<SigmoidLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kSigmoidGetInstance("nn.Sigmoid",
                                           SigmoidLayer::GetInstance);
}  // namespace kuiper_infer