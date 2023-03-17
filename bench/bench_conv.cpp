//
// Created by fushenshen on 2023/3/15.
//

#include <benchmark/benchmark.h>
#include "../source/layer/details/convolution.hpp"
#include "../source/layer/details/winograd.hpp"
#include "./data/tensor.hpp"

static void BM_32x16x512x512ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(16, 512, 512);
  input->Fill(1.f);

  const uint32_t kernel_count = 32;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(16, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_32x16x512x512ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(16, 512, 512);
  input->Fill(1.f);

  const uint32_t kernel_count = 32;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(16, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 16, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_32x32x320x320ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 320, 320);
  input->Fill(1.f);

  const uint32_t kernel_count = 32;
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

static void BM_32x32x320x320ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 320, 320);
  input->Fill(1.f);

  const uint32_t kernel_count = 32;
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

static void BM_64x64x80x80ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(64, 80, 80);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(64, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_64x64x80x80ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(64, 80, 80);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(64, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 64, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_32x32x80x80ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 80, 80);
  input->Fill(1.f);

  const uint32_t kernel_count = 32;
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

static void BM_32x32x80x80ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 80, 80);
  input->Fill(1.f);

  const uint32_t kernel_count = 32;
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

static void BM_64x16x40x40ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(16, 40, 40);
  input->Fill(1.f);

  const uint32_t kernel_count = 16;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(16, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_64x16x40x40ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(16, 40, 40);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(16, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 16, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_16x3x640x640ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(3, 640, 640);
  input->Fill(1.f);

  const uint32_t kernel_count = 16;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(3, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_16x3x640x640ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(3, 640, 640);
  input->Fill(1.f);

  const uint32_t kernel_count = 16;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(3, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 3, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}


static void BM_64x64x20x20ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(64, 20, 20);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(64, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_64x64x20x20ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(64, 20, 20);
  input->Fill(1.f);

  const uint32_t kernel_count = 64;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(64, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 64, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_256x128x80x80ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(128, 80, 80);
  input->Fill(1.f);

  const uint32_t kernel_count = 256;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(128, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  for (auto _ : state) {
    sftensor output;
    Convolution3x3s1(input, output, weights);
  }
}

static void BM_256x128x80x80ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(128, 80, 80);
  input->Fill(1.f);

  const uint32_t kernel_count = 256;
  std::vector<sftensor> weights(kernel_count);
  for (uint32_t k = 0; k < kernel_count; ++k) {
    sftensor weight = std::make_shared<ftensor>(128, 3, 3);
    weight->Rand();
    weights.at(k) = weight;
  }

  std::vector<sftensor> outputs(1);
  std::vector<sftensor> inputs;
  inputs.push_back(input);
  ConvolutionLayer conv_layer(kernel_count, 128, 3, 3, 0, 0, 1, 1, 1, false);
  conv_layer.set_weights(weights);
  for (auto _ : state) {
    conv_layer.Forward(inputs, outputs);
  }
}

static void BM_64x32x320x320ConvWinograd(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 320, 320);
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

static void BM_64x32x320x320ConvGemm(benchmark::State& state) {
  using namespace kuiper_infer;
  sftensor input = std::make_shared<ftensor>(32, 320, 320);
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


BENCHMARK(BM_32x16x512x512ConvWinograd)->Iterations(5);
BENCHMARK(BM_32x16x512x512ConvGemm)->Iterations(5);

BENCHMARK(BM_32x32x320x320ConvWinograd)->Iterations(5);
BENCHMARK(BM_32x32x320x320ConvGemm)->Iterations(5);

BENCHMARK(BM_64x64x80x80ConvWinograd)->Iterations(5);
BENCHMARK(BM_64x64x80x80ConvGemm)->Iterations(5);

BENCHMARK(BM_32x32x80x80ConvWinograd)->Iterations(5);
BENCHMARK(BM_32x32x80x80ConvGemm)->Iterations(5);

BENCHMARK(BM_64x16x40x40ConvWinograd)->Iterations(5);
BENCHMARK(BM_64x16x40x40ConvGemm)->Iterations(5);

BENCHMARK(BM_16x3x640x640ConvWinograd)->Iterations(5);
BENCHMARK(BM_16x3x640x640ConvGemm)->Iterations(5);

BENCHMARK(BM_64x64x20x20ConvWinograd)->Iterations(5);
BENCHMARK(BM_64x64x20x20ConvGemm)->Iterations(5);

BENCHMARK(BM_256x128x80x80ConvWinograd)->Iterations(5);
BENCHMARK(BM_256x128x80x80ConvGemm)->Iterations(5);

BENCHMARK(BM_64x32x320x320ConvWinograd)->Iterations(5);
BENCHMARK(BM_64x32x320x320ConvGemm)->Iterations(5);