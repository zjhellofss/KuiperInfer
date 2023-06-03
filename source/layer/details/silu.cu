//
// Created by fss on 22-12-25.
//

#include "layer/abstract/layer_factory.hpp"
#include "silu.hpp"
namespace kuiper_infer {

__global__ void SiLUForward(const int n, const float* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = in[index];
    out[index] = val / (1.f + expf(-val));
  }
}

SiLULayer::SiLULayer() : Layer("SiLU") {}

InferStatus SiLULayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the silu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the silu layer do "
                  "not match";
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
  SiLUForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, inputs.at(0)->gpu_data(), outputs.at(0)->gpu_data());
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus SiLULayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& silu_layer) {
  CHECK(op != nullptr) << "SiLU operator is nullptr";
  silu_layer = std::make_shared<SiLULayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kSiluGetInstance("nn.SiLU", SiLULayer::GetInstance);

}  // namespace kuiper_infer
