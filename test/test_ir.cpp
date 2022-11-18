//
// Created by fss on 22-11-13.
//

#include <gtest/gtest.h>
#include "parser/runtime_ir.hpp"
#include "tick.hpp"

TEST(test_ir, build) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet1819.pnnx.param", "tmp/resnet1819.pnnx.bin");
  bool init = graph.Init();
  CHECK(init == true);
  graph.Build();
  std::vector<std::shared_ptr<Tensor>> inputs;
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(6, 28, 28);

  std::vector<double> raw_inputs;
  const uint32_t total_size = 4704;
  for (uint32_t i = 0; i < total_size; ++i) {
    raw_inputs.push_back(i);
  }
  input->Fill(raw_inputs);
  inputs.push_back(input);
  inputs.push_back(input);
  graph.Forward(inputs, true);
}