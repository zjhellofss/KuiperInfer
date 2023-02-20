//
// Created by fss on 22-11-18.
//
#include "relu.hpp"
#include "layer/abstract/layer_factory.hpp"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.hpp"
#endif
namespace kuiper_infer {
InferStatus ReluLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of relu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const sftensor &input_data = inputs.at(i);
    const sftensor &output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input feature map of relu layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The input and output size is not adapting";
        return InferStatus::kInferFailedInputOutSizeAdaptingError;
      }
    }
  }

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(input == nullptr || !input->empty())
            << "The input feature map of relu layer is empty";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      DLOG(ERROR) << "The output size of relu is error";
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(output->shapes() == input->shapes())
            << "The output size of relu is error";
    const uint32_t size = input->size();
#pragma omp simd
    for (int j = 0; j < size; ++j) {
      float val = input->index(j);
      output->index(j) = val > 0.f ? val : 0.f;
    }
  }
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus ReluLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &relu_layer) {
  CHECK(op != nullptr) << "Relu operator is nullptr";
  relu_layer = std::make_shared<ReluLayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kReluGetInstance("nn.ReLU", ReluLayer::GetInstance);
}  // namespace kuiper_infer
