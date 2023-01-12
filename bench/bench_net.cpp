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

static void BM_Yolov5nano(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/yolo/yolov5n_small.pnnx.param",
                     "tmp/yolo/yolov5n_small.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 320, 320);
    input->Ones();
    inputs.push_back(input);
  }

  for (auto _ : state) {
    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
  }
}

static void BM_Yolov5s(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/yolo/demo/yolov5s_batch4.pnnx.param",
                     "tmp/yolo/demo/yolov5s_batch4.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 640, 640);
    input->Ones();
    inputs.push_back(input);
  }

  for (auto _ : state) {
    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
  }
}

//BENCHMARK(BM_Resnet18)->Iterations(kIterationNum);
//BENCHMARK(BM_Resnet18_Batch16)->Iterations(kIterationNum);
//BENCHMARK(BM_MobilenetV3)->Iterations(kIterationNum);
//BENCHMARK(BM_Yolov5nano)->Iterations(kIterationNum);
//BENCHMARK(BM_Yolov5s)->Iterations(kIterationNum);

BENCHMARK_MAIN();