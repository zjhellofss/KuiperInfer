//
// Created by fss on 23-6-14.
//

#include <benchmark/benchmark.h>
#include <armadillo>
#include "../source/layer/details/sse_mathfun.hpp"
#include "data/tensor.hpp"
#include "utils/math/arma_sse.hpp"
static void BM_ExpSimd(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  using namespace kuiper_infer::math;
  const uint32_t size = input->size();
  arma::fcube input_data = input->data();
  for (auto _ : state) {
    ArmaSigmoid(input_data, input_data);
  }
}

BENCHMARK(BM_ExpSimd)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExpSimd)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExpSimd)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);

static void BM_ExpArma(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
  for (auto _ : state) {
    const arma::fcube& input_data = input->data();
    const arma::fcube& input_data_exp = 1.f / (1.f + arma::exp(-input_data));
  }
}

BENCHMARK(BM_ExpArma)->Args({255, 80, 80})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExpArma)->Args({255, 40, 40})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ExpArma)->Args({255, 20, 20})->Unit(benchmark::kMillisecond);
