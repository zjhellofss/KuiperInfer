//
// Created by fss on 22-12-25.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"

TEST(test_net, forward_yolo1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/yolo/yolov5n.pnnx.param",
                     "tmp/yolo/yolov5n.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 640, 640);
    input->Ones();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, true);
  int d = 3;
}