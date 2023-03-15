//
// Created by fushenshen on 2023/3/15.
//

#include <benchmark/benchmark.h>
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/winograd.hpp"
#include "./data/tensor.hpp"

static void BM_128x512x512ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(128, 512, 512);
  input->Rand();
  sftensor weight = std::make_shared<ftensor>(128, 3, 3);
  weight->Rand();
  sftensor output;

  for (auto _ : state) {
    Convolution3x3s1(input, output, weight);
  }
}

static void BM_128x512x512ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(128, 512, 512);
  input->Rand();
  sftensor weight = std::make_shared<ftensor>(128, 3, 3);
  weight->Rand();
  sftensor output;

  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(1);
  std::vector<sftensor> weights(1);
  weights.at(0) = weight;
  // 设置权重
  ConvolutionLayer conv_layer(1, 128, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);

  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_16x32x32ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(16, 32, 32);
  input->Rand();
  sftensor weight = std::make_shared<ftensor>(16, 3, 3);
  weight->Rand();
  sftensor output;

  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(1);
  std::vector<sftensor> weights(1);
  weights.at(0) = weight;
  // 设置权重
  ConvolutionLayer conv_layer(1, 16, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);

  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_16x32x32ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(16, 32, 32);
  input->Rand();
  sftensor weight = std::make_shared<ftensor>(16, 3, 3);
  weight->Rand();
  sftensor output = std::make_shared<ftensor>(1, 30, 30);

  for (auto _ : state) {
    Convolution3x3s1(input, output, weight);
  }
}

static void BM_256x16x16ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(256, 16, 16);
  input->Rand();
  sftensor weight = std::make_shared<ftensor>(256, 3, 3);
  weight->Rand();
  sftensor output;

  std::vector<sftensor> inputs;
  inputs.push_back(input);
  std::vector<sftensor> outputs(1);
  std::vector<sftensor> weights(1);
  weights.at(0) = weight;
  // 设置权重
  ConvolutionLayer conv_layer(1, 256, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);

  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_256x16x16ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(256, 16, 16);
  input->Rand();
  sftensor weight = std::make_shared<ftensor>(256, 3, 3);
  weight->Rand();
  sftensor output;

  for (auto _ : state) {
    Convolution3x3s1(input, output, weight);
  }
}

BENCHMARK(BM_128x512x512ConvWinograd);
BENCHMARK(BM_128x512x512ConvGemm);

BENCHMARK(BM_16x32x32ConvWinograd);
BENCHMARK(BM_16x32x32ConvGemm);

BENCHMARK(BM_256x16x16ConvWinograd);
BENCHMARK(BM_256x16x16ConvGemm);
