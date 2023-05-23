//
// Created by fss on 22-11-18.
//
#include "layer/abstract/layer_factory.hpp"
#include "relu.hpp"

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
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the relu layer do "
                  "not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor& input_data = inputs.at(i);
    const sftensor& output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR)
          << "The input tensor array in the relu layer has an empty tensor "
          << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The input and output tensor shapes of the relu "
                      "layer do not match "
                   << i << " th";
        return InferStatus::kInferFailedInputOutSizeMatchError;
      }
    }
  }

  const float* inputs_data = inputs.at(0)->gpu_data();
  float* outputs_data = outputs.at(0)->gpu_data();

  uint32_t count = inputs.at(0)->size();

  CHECK(inputs_data == nullptr)
      << "The input tensor array in the relu layer has an empty tensor " ;


  if (outputs_data == nullptr) {
    DLOG(ERROR)
        << "The output tensor array in the relu layer has an empty tensor " << i
        << " th";
    std::shared_ptr<Tensor<float>> output =
        std::make_shared<Tensor<float>>(inputs[0]->shapes());
    outputs_data = output->gpu_data();
    outputs[0] = output;
  }

  ReLUForward<<<KUIPER_GET_BLOCKS(count), KUIPER_CUDA_NUM_THREADS>>>(
      count, inputs_data, outputs_data);

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
