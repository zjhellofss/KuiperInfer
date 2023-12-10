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

// Created by fss on 23-4-27.
#include <benchmark/benchmark.h>
#include "../source/layer/details/adaptive_avgpooling.hpp"
#include "../source/layer/details/cat.hpp"
#include "../source/layer/details/expression.hpp"
#include "../source/layer/details/flatten.hpp"
#include "../source/layer/details/hardsigmoid.hpp"
#include "../source/layer/details/hardswish.hpp"
#include "../source/layer/details/linear.hpp"
#include "../source/layer/details/maxpooling.hpp"
#include "../source/layer/details/relu.hpp"
#include "../source/layer/details/sigmoid.hpp"
#include "../source/layer/details/silu.hpp"
#include "../source/layer/details/softmax.hpp"
#include "../source/layer/details/upsample.hpp"
#include "../source/layer/details/view.hpp"

static void BM_Concat16to8(benchmark::State& state) {
  using namespace kuiper_infer;
  int input_size = 16;
  int output_size = 8;
  int input_channels = state.range(0);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_channels, state.range(1), state.range(2));
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);

  for (auto _ : state) {
    cat_layer.Forward(inputs, outputs);
  }
}

// 8 channel to 4 channel
BENCHMARK(BM_Concat16to8)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Concat16to8)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Concat16to8)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Concat16to8)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_Sigmoid(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  SigmoidLayer sigmoid_layer;
  for (auto _ : state) {
    sigmoid_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Sigmoid)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Sigmoid)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Sigmoid)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Sigmoid)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_HardSigmoid(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  HardSigmoid hardsigmoid_layer;
  for (auto _ : state) {
    hardsigmoid_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_HardSigmoid)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSigmoid)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSigmoid)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSigmoid)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_Linear(benchmark::State& state) {
  using namespace kuiper_infer;
  const int32_t in_features = (int32_t)state.range(0);
  const int32_t out_features = (int32_t)state.range(1);
  const int32_t in_dims = (int32_t)state.range(2);

  LinearLayer linear_layer(in_features, out_features, true);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear_layer.set_weights(weights);

  std::vector<float> bias(out_features, 3.f);
  linear_layer.set_bias(bias);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, in_dims, in_features);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output = std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  for (auto _ : state) {
    linear_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Linear)->Args({3, 32, 1000})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({32, 64, 1000})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({64, 128, 1000})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({128, 512, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({128, 1000, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({128, 2048, 512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Linear)->Args({512, 1024, 1000})->Unit(benchmark::kMillisecond);

static void BM_Expression(benchmark::State& state) {
  const int32_t channels = (int32_t)state.range(0);
  const int32_t rows = (int32_t)state.range(1);
  const int32_t cols = (int32_t)state.range(2);

  using namespace kuiper_infer;
  const std::string& str = "mul(add(@0,@1),add(@2,@3))";
  ExpressionLayer layer(str);
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(channels, rows, cols);
  input1->Fill(2.f);
  std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(channels, rows, cols);
  input2->Fill(3.f);

  std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(channels, rows, cols);
  input3->Fill(4.f);

  std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(channels, rows, cols);
  input4->Fill(4.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);
  inputs.push_back(input2);
  inputs.push_back(input3);
  inputs.push_back(input4);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  for (auto _ : state) {
    layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Expression)->Args({3, 320, 320});
BENCHMARK(BM_Expression)->Args({32, 160, 160});
BENCHMARK(BM_Expression)->Args({64, 80, 80});
BENCHMARK(BM_Expression)->Args({128, 40, 40});

static void BM_HardSwish(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  HardSwishLayer hardswish_layer;
  for (auto _ : state) {
    hardswish_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_HardSwish)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSwish)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSwish)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_HardSwish)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_SiLU(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  SiLULayer silu_layer;
  for (auto _ : state) {
    silu_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_SiLU)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiLU)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiLU)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiLU)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_ReLU(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  ReluLayer relu_layer;
  for (auto _ : state) {
    relu_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_ReLU)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReLU)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReLU)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReLU)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_MaxPooling_k3x3s1x1(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;

  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);
  for (auto _ : state) {
    max_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_MaxPooling_k3x3s1x1)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MaxPooling_k3x3s1x1)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MaxPooling_k3x3s1x1)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_MaxPooling_k3x3s1x1)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_View(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);

  ViewLayer view_layer({1, int(channels * rows * cols)});
  for (auto _ : state) {
    const auto status = view_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_View)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_View)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_View)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_View)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_Upsample(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t channels = state.range(0);
  uint32_t rows = state.range(1);
  uint32_t cols = state.range(2);

  UpSampleMode mode = UpSampleMode(state.range(3));
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  input->RandN();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  UpSampleLayer layer(3.f, 3.f, mode);

  for (auto _ : state) {
    const auto status = layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Upsample)->Args({3, 320, 320, 0})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Upsample)->Args({32, 160, 160, 0})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Upsample)->Args({64, 80, 80, 0})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Upsample)->Args({128, 40, 40, 0})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Upsample)->Args({3, 320, 320, 1})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Upsample)->Args({32, 160, 160, 1})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Upsample)->Args({64, 80, 80, 1})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Upsample)->Args({128, 40, 40, 1})->Unit(benchmark::kMillisecond);

static void BM_AdaptivePooling(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);

  const uint32_t output_h = input_h / 2;
  const uint32_t output_w = input_w / 2;

  std::vector<sftensor> inputs;
  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);
  std::vector<std::shared_ptr<Tensor<float>>> outputs(input_size);

  for (auto _ : state) {
    average_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_AdaptivePooling)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AdaptivePooling)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AdaptivePooling)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AdaptivePooling)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_Flatten(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);

  std::vector<sftensor> inputs;
  const uint32_t input_size = 1;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  FlattenLayer flatten_layer(1, 3);
  for (auto _ : state) {
    flatten_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Flatten)->Args({1, 224, 224})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Flatten)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);

static void BM_Softmax(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t input_c = state.range(0);

  std::vector<sftensor> inputs;
  const uint32_t input_size = 1;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(std::vector<uint32_t>{input_c});
    input->RandN();
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  SoftmaxLayer softmax_layer(-1);
  for (auto _ : state) {
    softmax_layer.Forward(inputs, outputs);
  }
}

static void BM_SoftmaxDim1Batch8(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);

  std::vector<sftensor> inputs;
  const uint32_t input_size = 8;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input =
        std::make_shared<Tensor<float>>(input_c, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(8);
  SoftmaxLayer softmax_layer(1);
  for (auto _ : state) {
    softmax_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_Softmax)->Args({32})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Softmax)->Args({512})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Softmax)->Args({1024})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Softmax)->Args({4096})->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SoftmaxDim1Batch8)->Args({1, 224, 224})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)->Args({8, 128, 128})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)->Args({3, 320, 320})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)->Args({32, 160, 160})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)->Args({64, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SoftmaxDim1Batch8)->Args({128, 40, 40})->Unit(benchmark::kMillisecond);
