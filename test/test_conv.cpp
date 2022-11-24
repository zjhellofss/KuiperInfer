//
// Created by fss on 22-11-22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parser/runtime_ir.hpp"
#include "data/load_data.hpp"

TEST(test_layer, check_dag) {
  using namespace kuiper_infer;
}

TEST(test_layer, forward_conv1_small) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/im2cols/conv1x1.pnnx.param", "tmp/im2cols/conv1x1.pnnx.bin");
  graph.Build();
  std::vector<std::shared_ptr<Tensor>> inputs;
  std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>(3, 5, 5);
  input_tensor->at(0) = CSVDataLoader::LoadData("./tmp/im2cols/1.csv");
  input_tensor->at(1) = CSVDataLoader::LoadData("./tmp/im2cols/2.csv");
  input_tensor->at(2) = CSVDataLoader::LoadData("./tmp/im2cols/3.csv");
  inputs.push_back(input_tensor);

  graph.Forward(inputs, true);
}

TEST(test_layer, forward_conv1_large) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/im2cols/conv1x1.pnnx.param", "tmp/im2cols/conv1x1.pnnx.bin");
  graph.Build();
  std::vector<std::shared_ptr<Tensor>> inputs;
  std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>(3, 512, 512);
  input_tensor->at(0) = CSVDataLoader::LoadData("./tmp/im2cols/1_large.csv");
  input_tensor->at(1) = CSVDataLoader::LoadData("./tmp/im2cols/2_large.csv");
  input_tensor->at(2) = CSVDataLoader::LoadData("./tmp/im2cols/3_large.csv");
  inputs.push_back(input_tensor);

  const auto &output = graph.Forward(inputs, false);
  ASSERT_EQ(output.size(), 1);
}