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
#include "simd.hpp"
#include <glog/logging.h>
#include <armadillo>
#include "utils/math/fmath.hpp"

namespace kuiper_infer {

namespace activation {

static void SigmoidSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index = 0;
  int64_t packet_size;
  int64_t in_size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 one = _mm256_set1_ps(1.f);
  __m256 zero = _mm256_setzero_ps();
  for (; index <= in_size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    p = _mm256_div_ps(one, _mm256_add_ps(one, fmath::exp_ps256(_mm256_sub_ps(zero, p))));
    _mm256_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE2__
  packet_size = 4;
  __m128 one128 = _mm_set1_ps(1.f);
  __m128 zero128 = _mm_setzero_ps();
  for (; index <= in_size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    p = _mm_div_ps(one128, _mm_add_ps(one128, fmath::exp_ps(_mm_sub_ps(zero128, p))));
    _mm_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
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
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_setzero_ps();
  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    __m256 value = _mm256_max_ps(zero, p);
    _mm256_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE__
  packet_size = 4;
  __m128 zero128 = _mm_setzero_ps();
  for (; index <= size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    __m128 value = _mm_max_ps(zero128, p);
    _mm_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      output->index(index) = std::max(value, 0.f);
      index += 1;
    }
  }
}

static void Relu6SSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_setzero_ps();
  __m256 six = _mm256_set1_ps(6.f);

  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    __m256 value = _mm256_min_ps(_mm256_max_ps(zero, p), six);
    _mm256_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE__
  packet_size = 4;
  __m128 zero128 = _mm_setzero_ps();
  __m128 six128 = _mm_set1_ps(6.f);
  for (; index <= size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    __m128 value = _mm_min_ps(_mm_max_ps(zero128, p), six128);
    _mm_storeu_ps(out_ptr, value);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      output->index(index) = std::min(std::max(value, 0.f), 6.f);
      index += 1;
    }
  }
}

static void SiluSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";
  int64_t index;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());
  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 one_256 = _mm256_set1_ps(1.f);
  __m256 zero_256 = _mm256_setzero_ps();

  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 p = _mm256_loadu_ps(in_ptr);
    p = _mm256_div_ps(p, _mm256_add_ps(one_256, fmath::exp_ps256(_mm256_sub_ps(zero_256, p))));
    _mm256_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE__
  packet_size = 4;
  __m128 one128 = _mm_set1_ps(1.f);
  __m128 zero128 = _mm_setzero_ps();

  for (; index <= size - packet_size; index += packet_size) {
    __m128 p = _mm_loadu_ps(in_ptr);
    p = _mm_div_ps(p, _mm_add_ps(one128, fmath::exp_ps(_mm_sub_ps(zero128, p))));
    _mm_storeu_ps(out_ptr, p);
    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      output->index(index) = value / (1.f + fmath::exp(-value));
      index += 1;
    }
  }
}

static void HardSwishSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";

  int64_t index;
  float threshold = 3.f;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());

  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_set1_ps(0.f);
  __m256 three = _mm256_set1_ps(threshold);
  __m256 six = _mm256_set1_ps(6.f);
  __m256 minus_three = _mm256_set1_ps(-threshold);
  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 x = _mm256_loadu_ps(in_ptr);

    __m256 le_branch = _mm256_cmp_ps(x, minus_three, _CMP_LE_OS);  // <= -3
    __m256 ge_branch = _mm256_cmp_ps(x, three, _CMP_GE_OS);        // >= 3
    __m256 mid_branch = _mm256_and_ps(_mm256_cmp_ps(x, minus_three, _CMP_GT_OS),
                                      _mm256_cmp_ps(x, three, _CMP_LT_OS));  // -3 < x < 3

    __m256 f1 = _mm256_and_ps(zero, le_branch);
    __m256 f2 = _mm256_and_ps(x, ge_branch);
    __m256 f3 =
        _mm256_and_ps(_mm256_div_ps(_mm256_mul_ps(x, _mm256_add_ps(x, three)), six), mid_branch);

    __m256 result = _mm256_add_ps(_mm256_add_ps(f1, f2), f3);
    _mm256_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE2__
  packet_size = 4;
  __m128 zero128 = _mm_set1_ps(0.f);
  __m128 three128 = _mm_set1_ps(threshold);
  __m128 six128 = _mm_set1_ps(6.f);
  __m128 minus_three128 = _mm_set1_ps(-threshold);
  for (; index <= size - packet_size; index += packet_size) {
    __m128 x = _mm_loadu_ps(in_ptr);

    __m128 le_branch = _mm_cmple_ps(x, minus_three128);  // <= -3
    __m128 ge_branch = _mm_cmpge_ps(x, three128);        // >= 3
    __m128 mid_branch =
        _mm_and_ps(_mm_cmpgt_ps(x, minus_three128), _mm_cmplt_ps(x, three128));  // -3 < x < 3

    __m128 f1 = _mm_and_ps(zero128, le_branch);
    __m128 f2 = _mm_and_ps(x, ge_branch);
    __m128 f3 = _mm_and_ps(_mm_div_ps(_mm_mul_ps(x, _mm_add_ps(x, three128)), six128), mid_branch);

    __m128 result = _mm_add_ps(_mm_add_ps(f1, f2), f3);
    _mm_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      float result = 0.f;
      if (value <= -3.f) {
        result = 0.f;
      } else if (value >= 3.f) {
        result = value;
      } else {
        result = value * (value + threshold) / 6;
      }
      output->index(index) = result;
      index += 1;
    }
  }
}

static void HardSigmoidSSE(sftensor input, sftensor output) {
  CHECK(input != nullptr && output != nullptr) << "The input or output tensor is empty.";
  CHECK(!input->empty() && !output->empty()) << "The input or output tensor is empty.";
  CHECK(input->size() == output->size()) << "The input and output sizes are not equal.";

  int64_t index;
  float threshold = 3.f;
  int64_t packet_size;
  int64_t size = static_cast<int64_t>(input->size());

  const float* in_ptr = input->raw_ptr();
  float* out_ptr = output->raw_ptr();
#ifdef __AVX2__
  packet_size = 8;
  __m256 zero = _mm256_set1_ps(0.f);
  __m256 one = _mm256_set1_ps(1.f);

  __m256 three = _mm256_set1_ps(threshold);
  __m256 six = _mm256_set1_ps(6.f);
  __m256 point_five = _mm256_set1_ps(0.5f);
  __m256 minus_three = _mm256_set1_ps(-threshold);
  for (index = 0; index <= size - packet_size; index += packet_size) {
    __m256 x = _mm256_loadu_ps(in_ptr);
    __m256 le_branch = _mm256_cmp_ps(x, minus_three, _CMP_LE_OS);  // <= -3
    __m256 ge_branch = _mm256_cmp_ps(x, three, _CMP_GE_OS);        // >= 3
    __m256 mid_branch = _mm256_and_ps(_mm256_cmp_ps(x, minus_three, _CMP_GT_OS),
                                      _mm256_cmp_ps(x, three, _CMP_LT_OS));  // -3 < x < 3

    __m256 f1 = _mm256_and_ps(zero, le_branch);
    __m256 f2 = _mm256_and_ps(one, ge_branch);
    __m256 f3 = _mm256_and_ps(_mm256_add_ps(_mm256_div_ps(x, six), point_five), mid_branch);

    __m256 result = _mm256_add_ps(_mm256_add_ps(f1, f2), f3);
    _mm256_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#ifdef __SSE2__
  packet_size = 4;
  __m128 zero128 = _mm_set1_ps(0.f);
  __m128 one128 = _mm_set1_ps(1.f);

  __m128 three128 = _mm_set1_ps(threshold);
  __m128 six128 = _mm_set1_ps(6.f);
  __m128 point_five128 = _mm_set1_ps(0.5f);
  __m128 minus_three128 = _mm_set1_ps(-threshold);
  for (; index <= size - packet_size; index += packet_size) {
    __m128 x = _mm_loadu_ps(in_ptr);
    __m128 le_branch = _mm_cmp_ps(x, minus_three128, _CMP_LE_OS);  // <= -3
    __m128 ge_branch = _mm_cmp_ps(x, three128, _CMP_GE_OS);        // >= 3
    __m128 mid_branch = _mm_and_ps(_mm_cmp_ps(x, minus_three128, _CMP_GT_OS),
                                   _mm_cmp_ps(x, three128, _CMP_LT_OS));  // -3 < x < 3

    __m128 f1 = _mm_and_ps(zero128, le_branch);
    __m128 f2 = _mm_and_ps(one128, ge_branch);
    __m128 f3 = _mm_and_ps(_mm_add_ps(_mm_div_ps(x, six128), point_five128), mid_branch);

    __m128 result = _mm_add_ps(_mm_add_ps(f1, f2), f3);
    _mm_storeu_ps(out_ptr, result);

    in_ptr += packet_size;
    out_ptr += packet_size;
  }
#endif
#endif
  if (index < size) {
    while (index < size) {
      float value = input->index(index);
      float result = 0.f;
      if (value <= -3.f) {
        result = 0.f;
      } else if (value >= 3.f) {
        result = 1.f;
      } else {
        result = value / 6.f + 0.5f;
      }
      output->index(index) = result;
      index += 1;
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
    case ActivationType::kActivationRelu6: {
      function = Relu6SSE;
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
    case ActivationType::kActivationHardSwish: {
      function = HardSwishSSE;
      return function;
    }
    case ActivationType::kActivationHardSigmoid: {
      function = HardSigmoidSSE;
      return function;
    }
    default: {
      LOG(FATAL) << "Unknown SSE activation type: " << int32_t(act_type);
    }
  }
}
}  // namespace activation
}  // namespace kuiper_infer