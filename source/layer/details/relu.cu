//
// Created by fss on 22-11-18.
//

#include "layer/abstract/layer_factory.hpp"
#include "relu.hpp"
#include "utils/gpu_utils.cuh"
namespace kuiper_infer {

__global__ void ReLUForward(const int n, const float* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) { out[index] = in[index] > 0 ? in[index] : 0; }
}

InferStatus ReluLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the relu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
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

  uint32_t count = inputs.at(0)->size();
  float* bottom_data = inputs.at(0)->gpu_data();
  float* top_data = outputs.at(0)->gpu_data();

  ReLUForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus ReluLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& relu_layer) {
  CHECK(op != nullptr) << "Relu operator is nullptr";
  relu_layer = std::make_shared<ReluLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kReluGetInstance("nn.ReLU", ReluLayer::GetInstance);
}  // namespace kuiper_infer
