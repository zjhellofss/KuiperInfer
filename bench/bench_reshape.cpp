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

// Created by fss on 2023/3/20.
#include <benchmark/benchmark.h>
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"

static void BM_ReshapeRowMajor(benchmark::State& state) {
  using namespace kuiper_infer;
  std::vector<sftensor> tensors;
  const uint32_t batch_size = 8;
  for (uint32_t i = 0; i < batch_size; ++i) {
    tensors.push_back(TensorCreate<float>({32, 320, 320}));
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      auto tensor = tensors.at(i);
      tensor->Reshape({320, 320, 32}, true);
    }
  }
}

static void BM_ReshapeColMajor(benchmark::State& state) {
  using namespace kuiper_infer;
  std::vector<sftensor> tensors;
  const uint32_t batch_size = 8;
  for (uint32_t i = 0; i < batch_size; ++i) {
    tensors.push_back(TensorCreate<float>({32, 320, 320}));
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      auto tensor = tensors.at(i);
      tensor->Reshape({320, 320, 32}, false);
    }
  }
}

static void BM_FillRowMajor(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor tensor = TensorCreate<float>({32, 320, 320});
  std::vector<float> values(32 * 320 * 320, 1.f);
  for (auto _ : state) {
    tensor->Fill(values, true);
  }
}

BENCHMARK(BM_FillRowMajor)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReshapeRowMajor)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReshapeColMajor)->Unit(benchmark::kMillisecond);
