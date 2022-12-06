//
// Created by fss on 22-11-22.
//
#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"
#include "data/fast_copy.hpp"

static void BM_Copy1(benchmark::State &state) {
  using namespace kuiper_infer;
  const int size = 32 * 32 * 1024;
  std::vector<float> in(size);
  for (int i = 0; i < size; ++i) {
    in.at(i) = float(i);
  }

  std::vector<float> out(size);

  for (auto _ : state) {
    FastMemoryCopy(out.data(), in.data(), size);
  }
}

static void BM_Copy2(benchmark::State &state) {
  using namespace kuiper_infer;
  const int size = 32 * 32 * 1024;
  std::vector<float> in(size);
  for (int i = 0; i < size; ++i) {
    in.at(i) = float(i);
  }

  std::vector<float> out(size);

  for (auto _ : state) {
    memcpy(out.data(), in.data(), size * sizeof(float));
  }
}

//BENCHMARK(BM_Copy1);
//BENCHMARK(BM_Copy2);
