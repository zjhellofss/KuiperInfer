//
// Created by fushenshen on 2023/3/15.
//

#include <benchmark/benchmark.h>
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/winograd.hpp"

static void BM_Convolutionk3x3s1x1(benchmark::State& state) {
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

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({32, 3, 320, 320})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({64, 32, 160, 160})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({128, 64, 80, 80})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({256, 128, 40, 40})
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Convolutionk3x3s1x1)
    ->Args({512, 256, 20, 20})
    ->Unit(benchmark::kMillisecond);
