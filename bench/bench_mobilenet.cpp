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

// Created by fss on 22-11-22.
#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"

static void BM_MobilenetV3_Batch8_224x224(benchmark::State& state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile_batch8.pnnx.param", "tmp/mobilenet/mobile_batch8.bin");

  graph.Build();
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  const uint32_t batch_size = 8;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);
    inputs.push_back(input1);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  for (auto _ : state) {
    graph.Forward(false);
  }
}

BENCHMARK(BM_MobilenetV3_Batch8_224x224)->Unit(benchmark::kMillisecond);
