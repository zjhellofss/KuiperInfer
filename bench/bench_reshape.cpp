//
// Created by fss on 2023/3/20.
//
#include <benchmark/benchmark.h>
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"

static void BM_ReshapeRowMajor(benchmark::State& state) {
  using namespace kuiper_infer;
  std::vector<sftensor> tensors;
  const uint32_t batch_size = 8;
  for (uint32_t i = 0; i < batch_size; ++i) {
    tensors.push_back(TensorCreate(32, 320, 320));
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      auto tensor = tensors.at(i);
      tensor->Reshape({320, 320, 32}, true);
    }
  }
}

static void BM_ReshapeColMajor(benchmark::State& state) {
  using namespace kuiper_infer;
  std::vector<sftensor> tensors;
  const uint32_t batch_size = 8;
  for (uint32_t i = 0; i < batch_size; ++i) {
    tensors.push_back(TensorCreate(32, 320, 320));
  }
  for (auto _ : state) {
    for (uint32_t i = 0; i < batch_size; ++i) {
      auto tensor = tensors.at(i);
      tensor->Reshape({320, 320, 32}, false);
    }
  }
}

BENCHMARK(BM_ReshapeRowMajor)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_ReshapeColMajor)->Unit(benchmark::kMillisecond);