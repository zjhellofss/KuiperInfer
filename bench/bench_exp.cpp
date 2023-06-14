//
// Created by fss on 23-6-14.
//

#include <benchmark/benchmark.h>
#include <armadillo>
#include "../source/layer/details/sse_mathfun.hpp"
#include "data/tensor.hpp"

static void BM_ExpSimd(benchmark::State& state) {
  using namespace kuiper_infer;
  uint32_t input_c = state.range(0);
  uint32_t input_h = state.range(1);
  uint32_t input_w = state.range(2);
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  const uint32_t size = input->size();
  for (auto _ : state) {
    float* ptr = input->raw_ptr();
    __m128 _one1 = _mm_set1_ps(1.f);
    __m128 _one2 = _mm_set1_ps(1.f);
    __m128 _zero = _mm_setzero_ps();
    const uint32_t packet_size = 4;
    uint32_t j = 0;
    for (j = 0; j < size - 3; j += packet_size) {
      __m128 _p = _mm_load_ps(ptr);
      _p = _mm_div_ps(_one1, _mm_add_ps(_one2, exp_ps(_mm_sub_ps(_zero, _p))));
      _mm_store_ps(ptr, _p);
      ptr += packet_size;
    }
    if (j < size) {
      while (j < size) {
        float value = input->index(j);
        input->index(j) = 1.f / (1.f + expf(-value));
        j += 1;
      }
    }
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
