//
// Created by fss on 23-2-2.
//
#include <benchmark/benchmark.h>
#include "runtime/runtime_ir.hpp"
const static int kIterationNum = 5;

static void BM_Yolov5nano_Batch4_320x320(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/yolo/demo/yolov5n_batch4.pnnx.param",
                     "tmp/yolo/demo/yolov5n_batch4.pnnx.bin");

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

static void BM_Yolov5s_Batch4_640x640(benchmark::State &state) {
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

static void BM_Yolov5s_Batch8_640x640(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/yolo/demo/yolov5s_batch8.pnnx.param",
                     "tmp/yolo/demo/yolov5s_batch8.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 8;
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

BENCHMARK(BM_Yolov5nano_Batch4_320x320);
BENCHMARK(BM_Yolov5s_Batch4_640x640);
BENCHMARK(BM_Yolov5s_Batch8_640x640);