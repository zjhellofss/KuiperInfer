//
// Created by fss on 23-1-12.
//
#include <benchmark/benchmark.h>
#include "data/tensor.hpp"
#if __SSE2__
#include <emmintrin.h>
#endif

static void BM_CopyTensor1(benchmark::State &state) {
  using namespace kuiper_infer;
  uint32_t batch_size = state.range(0);
  std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<Tensor<float>>(3, 320, 320);
    inputs.at(i)->Rand();
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    outputs.at(i) = std::make_shared<Tensor<float>>(3, 320, 320);
    outputs.at(i)->Rand();
  }

  for (auto _ : state) {
    for (int i = 0; i < batch_size; ++i) {
      outputs.at(i)->set_data(inputs.at(i)->data());
    }
  }
}

static void BM_CopyTensor2(benchmark::State &state) {
  using namespace kuiper_infer;
  uint32_t batch_size = state.range(0);
  std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<Tensor<float>>(8, 320, 320);
    inputs.at(i)->Rand();
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    outputs.at(i) = std::make_shared<Tensor<float>>(8, 320, 320);
    outputs.at(i)->Rand();
  }

  for (auto _ : state) {
    for (int i = 0; i < batch_size; ++i) {
      float *desc_ptr = (float *) outputs.at(i)->raw_ptr();
      float *src_ptr = (float *) inputs.at(i)->raw_ptr();
      uint32_t size = outputs.at(i)->size();
      for (int j = 0; j < size - 3; j += 4) {
        __m128 p = _mm_load_ps(src_ptr);
        _mm_store_ps(desc_ptr, p);
        src_ptr += 4;
        desc_ptr += 4;
      }
    }
  }
}

static void BM_CopyTensor1OMP(benchmark::State &state) {
  using namespace kuiper_infer;
  uint32_t batch_size = state.range(0);
  std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<Tensor<float>>(8, 320, 320);
    inputs.at(i)->Rand();
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    outputs.at(i) = std::make_shared<Tensor<float>>(8, 320, 320);
    outputs.at(i)->Rand();
  }

  for (auto _ : state) {
#pragma omp parallel for num_threads(batch_size)
    for (int i = 0; i < batch_size; ++i) {
      uint32_t copy_size = inputs.at(i)->size();
      uint32_t dest_size = outputs.at(i)->size();
      memcpy((float *) outputs.at(i)->raw_ptr(), (float *) inputs.at(i)->raw_ptr(), sizeof(float) * copy_size);
    }
  }
}

static void BM_CopyTensor2OMP(benchmark::State &state) {
  using namespace kuiper_infer;
  uint32_t batch_size = state.range(0);
  std::vector<std::shared_ptr<Tensor<float>>> inputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<Tensor<float>>(3, 320, 320);
    inputs.at(i)->Rand();
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    outputs.at(i) = std::make_shared<Tensor<float>>(3, 320, 320);
    outputs.at(i)->Rand();
  }

  for (auto _ : state) {
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < batch_size; ++i) {
      float *desc_ptr = (float *) outputs.at(i)->raw_ptr();
      float *src_ptr = (float *) inputs.at(i)->raw_ptr();
      uint32_t size = outputs.at(i)->size();
#if __SSE2__
      for (int j = 0; j < size - 3; j += 4) {
        _mm_store_ps(desc_ptr, _mm_load_ps(src_ptr));
        src_ptr += 4;
        desc_ptr += 4;
      }
#endif
    }
  }
}

BENCHMARK(BM_CopyTensor1)->Args({4});
BENCHMARK(BM_CopyTensor1)->Args({8});
BENCHMARK(BM_CopyTensor1)->Args({16});
BENCHMARK(BM_CopyTensor1)->Args({32});

BENCHMARK(BM_CopyTensor1OMP)->Args({4});
BENCHMARK(BM_CopyTensor1OMP)->Args({8});
BENCHMARK(BM_CopyTensor1OMP)->Args({16});
BENCHMARK(BM_CopyTensor1OMP)->Args({32});

BENCHMARK(BM_CopyTensor2)->Args({4});
BENCHMARK(BM_CopyTensor2)->Args({8});
BENCHMARK(BM_CopyTensor2)->Args({16});
BENCHMARK(BM_CopyTensor2)->Args({32});

BENCHMARK(BM_CopyTensor2OMP)->Args({4});
BENCHMARK(BM_CopyTensor2OMP)->Args({8});
BENCHMARK(BM_CopyTensor2OMP)->Args({16});
BENCHMARK(BM_CopyTensor2OMP)->Args({32});