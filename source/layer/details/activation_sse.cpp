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

//
// Created by fss on 23-7-26.
//
#include "activation_sse.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

namespace activation {

static void SigmoidSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr)
      << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty())
      << "The input or output tensor is empty.";
  CHECK(input->size() == output->size())
      << "The input and output sizes are not equal.";
  int32_t index = 0;
  int32_t packet_size = 4;
  int32_t in_size = static_cast<int32_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 _one = _mm256_set1_ps(1.f);
  __m256 _zero = _mm256_setzero_ps();
  for (; index <= in_size - packet_size; index += packet_size) {
    __m256 _p = _mm256_loadu_ps(in_ptr);
    _p = _mm256_div_ps(
        _one, _mm256_add_ps(_one, fmath::exp_ps256(_mm256_sub_ps(_zero, _p))));
    _mm256_storeu_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#elif __SSE2__
  __m128 _one = _mm_set1_ps(1.f);
  __m128 _zero = _mm_setzero_ps();
  for (; index <= in_size - packet_size; index += packet_size) {
    __m128 _p = _mm_load_ps(in_ptr);
    _p = _mm_div_ps(_one,
                    _mm_add_ps(_one, fmath::exp_ps(_mm_sub_ps(_zero, _p))));
    _mm_store_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
  if (index < in_size) {
    while (index < in_size) {
      float value = input->index(index);
      output->index(index) = 1 / (1.f + fmath::exp(-value));
      index += 1;
    }
  }
}

static void ReluSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr)
      << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty())
      << "The input or output tensor is empty.";
  CHECK(input->size() == output->size())
      << "The input and output sizes are not equal.";
  int32_t j = 0;
  int32_t packet_size = 4;
  int32_t size = static_cast<int32_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 _zero = _mm256_setzero_ps();
  for (j = 0; j <= size - packet_size; j += packet_size) {
    __m256 _p = _mm256_loadu_ps(in_ptr);
    __m256 _value = _mm256_max_ps(_zero, _p);
    _mm256_storeu_ps(out_ptr, _value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#elif __SSE__
  __m128 _zero = _mm_setzero_ps();
  for (j = 0; j <= size - packet_size; j += packet_size) {
    __m128 _p = _mm_load_ps(in_ptr);
    __m128 _value = _mm_max_ps(_zero, _p);
    _mm_store_ps(out_ptr, _value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
  if (j < size) {
    while (j < size) {
      float value = input->index(j);
      output->index(j) = std::max(value, 0.f);
      j += 1;
    }
  }
}

static void SiluSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr)
      << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty())
      << "The input or output tensor is empty.";
  CHECK(input->size() == output->size())
      << "The input and output sizes are not equal.";

  int32_t j = 0;
  int32_t packet_size = 4;
  int32_t size = static_cast<int32_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 _one = _mm256_set1_ps(1.f);
  __m256 _zero = _mm256_setzero_ps();

  for (j = 0; j <= size - packet_size; j += packet_size) {
    __m256 _p = _mm256_loadu_ps(in_ptr);
    _p = _mm256_div_ps(
        _p, _mm256_add_ps(_one, fmath::exp_ps256(_mm256_sub_ps(_zero, _p))));
    _mm256_storeu_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#else
  __m128 _one = _mm_set1_ps(1.f);
  __m128 _zero = _mm_setzero_ps();

  for (j = 0; j <= size - packet_size; j += packet_size) {
    __m128 _p = _mm_loadu_ps(in_ptr);
    _p = _mm_div_ps(_p, _mm_add_ps(_one, fmath::exp_ps(_mm_sub_ps(_zero, _p))));
    _mm_storeu_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
  if (j < size) {
    while (j < size) {
      float value = input->index(j);
      output->index(j) = value / (1.f + fmath::exp(-value));
      j += 1;
    }
  }
}

ActivationFunc ApplySSEActivation(ActivationType act_type) {
  ActivationFunc function;
  switch (act_type) {
    case ActivationType::kActivationRelu: {
      function = ReluSSE;
      return function;
    }
    case ActivationType::kActivationSigmoid: {
      function = SigmoidSSE;
      return function;
    }
    case ActivationType::kActivationSilu: {
      function = SiluSSE;
      return function;
    }
    default: {
      LOG(FATAL) << "Unknown SSE activation type: " << int(act_type);
    }
  }
}
}  // namespace activation
}  // namespace kuiper_infer