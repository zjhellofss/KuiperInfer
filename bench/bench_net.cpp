//
// Created by fss on 22-11-22.
//
#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"
const int kIterationNum = 5;

static void BM_Resnet18(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/resnet18_batch8.pnnx.param",
                     "tmp/resnet/resnet18_batch8.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  const uint32_t batch_size = 8;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);
    inputs.push_back(input1);
  }

  for (auto _ : state) {
    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  }
}

static void BM_Resnet18_Batch16(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/resnet18_batch16.pnnx.param",
                     "tmp/resnet/resnet18_batch16.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t batch_size = 16;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);
    inputs.push_back(input1);
  }

  for (auto _ : state) {
    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  }
}

static void BM_MobilenetV3(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile_batch8.pnnx.param",
                     "tmp/mobilenet/mobile_batch8.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  const uint32_t batch_size = 8;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);
    inputs.push_back(input1);
  }

  for (auto _ : state) {
    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  }
}

BENCHMARK(BM_Resnet18)->Iterations(kIterationNum);
BENCHMARK(BM_Resnet18_Batch16)->Iterations(kIterationNum);
BENCHMARK(BM_MobilenetV3)->Iterations(kIterationNum);
