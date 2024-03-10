#include <benchmark/benchmark.h>
#include <cstring>
#include <ctime>
#include <memory>
//
// Created by fss on 24-2-15.
//
#include "../demos/llama2/llama_chat.hpp"
#include "../source/layer/details/rms_norm.hpp"
#include "data/tensor.hpp"

static void BM_RMSNorm(benchmark::State& state) {
  using namespace kuiper_infer;
  int input_size = state.range(0);

  std::vector<std::shared_ptr<Tensor<float>>> input_tensors;
  std::vector<std::shared_ptr<Tensor<float>>> output_tensors;

  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(input_size);
  input_tensor->RandN();
  input_tensors.push_back(input_tensor);

  std::vector<sftensor> weight_tensors;
  weight_tensors.push_back(input_tensor);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(input_size);

  std::shared_ptr<Tensor<float>> output_tensor = std::make_shared<Tensor<float>>(input_size);
  output_tensors.push_back(output_tensor);

  RMSNormLayer rms;
  rms.set_weights(weight_tensors);
  for (auto _ : state) {
    rms.Forward(input_tensors, output_tensors);
  }
}

BENCHMARK(BM_RMSNorm)->Unit(benchmark::kMillisecond)->Arg(128)->Iterations(3);
BENCHMARK(BM_RMSNorm)->Unit(benchmark::kMillisecond)->Arg(512)->Iterations(3);
BENCHMARK(BM_RMSNorm)->Unit(benchmark::kMillisecond)->Arg(1024)->Iterations(3);
BENCHMARK(BM_RMSNorm)->Unit(benchmark::kMillisecond)->Arg(4096)->Iterations(3);
