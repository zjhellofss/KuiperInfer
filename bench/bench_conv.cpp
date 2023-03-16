//
// Created by fushenshen on 2023/3/15.
//

#include <benchmark/benchmark.h>
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/winograd.hpp"
#include "./data/tensor.hpp"

static void BM_64x32x512x512ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 512, 512);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(32, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_64x32x512x512ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 512, 512);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(32, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 32, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

BENCHMARK(BM_64x32x512x512ConvWinograd);
BENCHMARK(BM_64x32x512x512ConvGemm);
