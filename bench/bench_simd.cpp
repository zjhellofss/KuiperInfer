//
// Created by fss on 23-6-14.
//

#include <benchmark/benchmark.h>
#include <armadillo>
#include "../source/layer/details/simd.hpp"
static void BM_SigmoidSimd(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  using namespace kuiper_infer::activation;
  for (auto _ : state) {
    ApplySSEActivation(ActivationType::kActivationSigmoid)(input, input);
  }
}

BENCHMARK(BM_SigmoidSimd)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SigmoidSimd)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SigmoidSimd)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);

static void BM_SigmoidArma(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  input->RandN();
  for (auto _ : state) {
    arma::fcube input_data = input->data();
    input_data = 1.f / (1.f + arma::exp(-input_data));
  }
}

BENCHMARK(BM_SigmoidArma)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SigmoidArma)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SigmoidArma)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);

static void BM_Relu(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  input->RandN();
  using namespace kuiper_infer::activation;
  for (auto _ : state) {
    for (uint32_t j = 0; j < input->size(); ++j) {
      float value = input->index(j);
      input->index(j) = value > 0.f ? value : 0.f;
    }
  }
}

BENCHMARK(BM_Relu)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Relu)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Relu)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);

static void BM_ReluSimd(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  input->RandN();
  using namespace kuiper_infer::activation;
  for (auto _ : state) {
    ApplySSEActivation(ActivationType::kActivationRelu)(input, input);
  }
}

BENCHMARK(BM_ReluSimd)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReluSimd)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReluSimd)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);

static void BM_SiluArma(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  input->RandN();
  using namespace kuiper_infer::activation;
  for (auto _ : state) {
    arma::fcube input_data = input->data();
    input_data = input_data / (1.f + arma::exp(-input_data));
  }
}

BENCHMARK(BM_SiluArma)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiluArma)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiluArma)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);

static void BM_SiluSimd(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  input->RandN();
  using namespace kuiper_infer::activation;
  for (auto _ : state) {
    ApplySSEActivation(ActivationType::kActivationSilu)(input, input);
  }
}

BENCHMARK(BM_SiluSimd)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiluSimd)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_SiluSimd)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);