// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-13.

#include "sigmoid.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.hpp"
#endif

namespace kuiper_infer {

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

  const uint32_t batch_size = inputs.size();
#pragma omp parallel for num_threads(batch_size)
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input == nullptr || !input->empty())
        << "The input tensor array in the sigmoid layer has an empty tensor "
        << i << "th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }

    CHECK(output->shapes() == input->shapes())
        << "The input and output tensor shapes of the sigmoid layer do not "
           "match "
        << i << " th";
#if __SSE2__
    const uint32_t size = input->size();
    float* in_ptr = input->raw_ptr();
    float* out_ptr = output->raw_ptr();
    __m128 _one = _mm_set1_ps(1.f);
    __m128 _zero = _mm_setzero_ps();
    const uint32_t packet_size = 4;
    uint32_t j = 0;
    for (j = 0; j < size - 3; j += packet_size) {
      __m128 _p = _mm_load_ps(in_ptr);
      _p = _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
      _mm_store_ps(out_ptr, _p);
      in_ptr += packet_size;
      out_ptr += packet_size;
    }
    if (j < size) {
      while (j < size) {
        float value = input->index(j);
        output->index(j) = 1 / (1.f + expf(-value));
        j += 1;
      }
    }
#else
    output->set_data(1.f / (1.f + arma::exp(-input->data())));
#endif
  }
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