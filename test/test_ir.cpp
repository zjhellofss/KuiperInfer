//
// Created by fss on 22-11-13.
//

#include <gtest/gtest.h>
#include "parser/runtime_ir.hpp"
#include "tick.hpp"

TEST(test_ir, build) {
  using namespace kuiper_infer;

  RuntimeGraph graph("/home/fss/code/kuiper_infer/tmp/resnet181912.pnnx.param",
                     "/home/fss/code/kuiper_infer/tmp/resnet181912.pnnx.bin");
  bool init = graph.Init();
  CHECK(init == true);
  graph.Build();
  std::vector<std::shared_ptr<Tensor>> inputs;
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(3, 512,512);
  std::vector<double> input_data;

  input->Fill(1.);
  inputs.push_back(input);
  graph.Forward(inputs, true);
}