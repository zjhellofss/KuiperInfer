//
// Created by fss on 22-12-25.
//

#include "silu.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "tick.hpp"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.hpp"
#endif
namespace kuiper_infer {

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

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<ftensor>& input_data = inputs.at(i);
    const std::shared_ptr<ftensor>& output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR)
          << "The input tensor array in the silu layer has an empty tensor "
          << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (input_data->shapes() != output_data->shapes()) {
        LOG(ERROR) << "The input and output tensor shapes of the silu "
                      "layer do not match "
                   << i << " th";
        return InferStatus::kInferFailedInputOutSizeMatchError;
      }
    }
  }

#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input == nullptr || !input->empty())
        << "The input tensor array in the silu layer has an empty tensor " << i
        << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the silu layer do not match "
        << i << " th";
    uint32_t size = output->size();
#if __SSE2__
    float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();
    __m128 _one = _mm_set1_ps(1.f);
    __m128 _zero = _mm_setzero_ps();
    const uint32_t packet_size = 4;
    uint32_t j = 0;
    for (j = 0; j < size - 3; j += packet_size) {
      __m128 _p = _mm_load_ps(in_ptr);
      _p = _mm_div_ps(_p, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
      _mm_store_ps(out_ptr, _p);
      in_ptr += packet_size;
      out_ptr += packet_size;
    }
    if (j < size) {
      while (j < size) {
        float value = input->index(j);
        output->index(j) = value / (1.f + expf(-value));
        j += 1;
      }
    }
#else
    for (uint32_t j = 0; j < input->size(); ++j) {
      float value = input->index(j);
      output->index(j) = value / (1.f + expf(-value));
    }
#endif
  }
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
