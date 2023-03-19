//
// Created by fushenshen on 2023/3/15.
//

#include <benchmark/benchmark.h>
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/winograd.hpp"

static void BM_ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t kernel_count = state.range(0);
  uint32_t channels = state.range(1);
  uint32_t rows = state.range(2);
  uint32_t cols = state.range(3);
  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(channels, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;

  uint32_t kernel_count = state.range(0);
  uint32_t channels = state.range(1);
  uint32_t rows = state.range(2);
  uint32_t cols = state.range(3);

  sftensor input = std::make_shared<ftensor>(channels, rows, cols);
  input->Fill(1.f);

  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(channels, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, channels, 3, 3, 0, 0, 1, 1, 1,
                              false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_ConvWinograd)->Args({32, 3, 640, 640});
BENCHMARK(BM_ConvGemm)->Args({32, 3, 640, 640});

BENCHMARK(BM_ConvWinograd)->Args({32, 32,160,160});
BENCHMARK(BM_ConvGemm)->Args({32, 32,160,160});

BENCHMARK(BM_ConvWinograd)->Args({64, 32, 320, 320});
BENCHMARK(BM_ConvGemm)->Args({64, 32, 320, 320});