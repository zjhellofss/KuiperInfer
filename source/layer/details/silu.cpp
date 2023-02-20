//
// Created by fss on 22-12-25.
//

#include "silu.hpp"
#include "tick.hpp"
#include "layer/abstract/layer_factory.hpp"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.hpp"
#endif
namespace kuiper_infer {

SiLULayer::SiLULayer() : Layer("SiLU") {

}

InferStatus SiLULayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                               std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of silu layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output size of silu layer is not adapting";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<ftensor> &input_data = inputs.at(i);
    const std::shared_ptr<ftensor> &output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input feature map of silu layer is empty";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The output size of silu layer is not adapting";
        return InferStatus::kInferFailedOutputSizeError;
      }
    }
  }

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>> &input = inputs.at(i);
    CHECK(input == nullptr || !input->empty()) << "The input feature map of silu layer is empty!";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes()) << "The output size of silu layer is error";
    const uint32_t size = input->size();
#pragma omp simd
    for (int j = 0; j < size; ++j) {
      float val = input->index(j);
      output->index(j) = val / (1.f + expf(-val));
    }
//    output->set_data(input->data());
//    output->Transform([](const float value) {
//      return value / (1.f + expf(-value));
//    });
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus SiLULayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                std::shared_ptr<Layer> &silu_layer) {
  CHECK(op != nullptr) << "SiLU operator is nullptr";
  silu_layer = std::make_shared<SiLULayer>();
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kSiluGetInstance("nn.SiLU", SiLULayer::GetInstance);

}
