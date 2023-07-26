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
#include "utils/math/arma_sse.hpp"
#include <glog/logging.h>

namespace kuiper_infer {

namespace math {
void ArmaSigmoid(const arma::fcube& input_data, arma::fcube& output_data) {
  CHECK(!input_data.empty() && !output_data.empty())
      << "The input or output tensor is empty.";
  const uint32_t in_size = input_data.size();
  const uint32_t out_size = output_data.size();
  CHECK(in_size == out_size) << "The input and output sizes are not equal.";
#ifdef __SSE2__
  int32_t packet_size = 4;
  int32_t index = 0;
  const float* in_ptr = input_data.memptr();
  float* out_ptr = output_data.memptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 _one = _mm256_set1_ps(1.f);
  __m256 _zero = _mm256_setzero_ps();
  for (; index <= (int32_t)in_size - packet_size; index += packet_size) {
    __m256 _p = _mm256_loadu_ps(in_ptr);
    _p = _mm256_div_ps(
        _one, _mm256_add_ps(_one, fmath::exp_ps256(_mm256_sub_ps(_zero, _p))));
    _mm256_storeu_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#else
  __m128 _one = _mm_set1_ps(1.f);
  __m128 _zero = _mm_setzero_ps();
  for (; index <= (int32_t)in_size - packet_size; index += packet_size) {
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
      float value = input_data.at(index);
      output_data.at(index) = 1 / (1.f + expf(-value));
      index += 1;
    }
  }
#else
  output_data = 1.f / (1.f + arma::exp(-input_data));
#endif
}

void ArmaReLU(const arma::fcube& input_data, arma::fcube& output_data) {
  CHECK(!input_data.empty() && !output_data.empty())
      << "The input or output tensor is empty.";
  CHECK(input_data.size() == output_data.size())
      << "The input and output sizes are not equal.";
#ifndef __SSE2__
  for (uint32_t j = 0; j < input_data.size(); ++j) {
    float value = input_data.at(j);
    output_data.at(j) = value > 0.f ? value : 0.f;
  }
#else
  int32_t j = 0;
  int32_t packet_size = 4;
  const uint32_t size = output_data.size();
#ifdef __AVX2__
  packet_size = 8;
  const float* in_ptr = input_data.memptr();
  float* out_ptr = output_data.memptr();
  __m256 _zero = _mm256_setzero_ps();
  for (j = 0; j <= (int32_t)size - packet_size; j += packet_size) {
    __m256 _p = _mm256_loadu_ps(in_ptr);
    __m256 _value = _mm256_max_ps(_zero, _p);
    _mm256_storeu_ps(out_ptr, _value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#else
  const float* in_ptr = input_data.memptr();
  float* out_ptr = output_data.memptr();
  __m128 _zero = _mm_setzero_ps();
  for (j = 0; j <= (int32_t)size - packet_size; j += packet_size) {
    __m128 _p = _mm_load_ps(in_ptr);
    __m128 _value = _mm_max_ps(_zero, _p);
    _mm_store_ps(out_ptr, _value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
  if (j < size) {
    while (j < size) {
      float value = input_data.at(j);
      output_data.at(j) = value > 0.f ? value : 0.f;
      j += 1;
    }
  }
#endif
}

void ArmaSiLU(const arma::fcube& input_data, arma::fcube& output_data) {
  CHECK(!input_data.empty() && !output_data.empty())
      << "The input or output tensor is empty.";
  CHECK(input_data.size() == output_data.size())
      << "The input and output sizes are not equal.";
#ifndef __SSE2__
  output_data = input_data / (1 + arma::exp(-input_data));
#else
  int32_t j = 0;
  int32_t packet_size = 4;
  const uint32_t size = output_data.size();
#ifdef __AVX2__
  packet_size = 8;
  const float* in_ptr = input_data.memptr();
  float* out_ptr = output_data.memptr();
  __m256 _one = _mm256_set1_ps(1.f);
  __m256 _zero = _mm256_setzero_ps();

  for (j = 0; j <= (int32_t)size - packet_size; j += packet_size) {
    __m256 _p = _mm256_loadu_ps(in_ptr);
    _p = _mm256_div_ps(
        _p, _mm256_add_ps(_one, fmath::exp_ps256(_mm256_sub_ps(_zero, _p))));
    _mm256_storeu_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#else
  const float* in_ptr = input_data.memptr();
  float* out_ptr = output_data.memptr();
  __m128 _one = _mm_set1_ps(1.f);
  __m128 _zero = _mm_setzero_ps();

  for (j = 0; j <= (int32_t)size - packet_size; j += packet_size) {
    __m128 _p = _mm_loadu_ps(in_ptr);
    _p = _mm_div_ps(_p, _mm_add_ps(_one, fmath::exp_ps(_mm_sub_ps(_zero, _p))));
    _mm_storeu_ps(out_ptr, _p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
  if (j < size) {
    while (j < size) {
      float value = input_data.at(j);
      output_data.at(j) = value / (1.f + expf(-value));
      j += 1;
    }
  }
#endif
}

}  // namespace math
}  // namespace kuiper_infer
