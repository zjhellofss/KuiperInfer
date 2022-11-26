//
// Created by fss on 22-11-22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "parser/runtime_ir.hpp"
#include "data/load_data.hpp"

TEST(test_layer, forward_small_map_pool) {
  using namespace kuiper_infer;
  RuntimeGraph
      graph("tmp/small_graph_pool/resnet_small_graph1.pnnx.param", "tmp/small_graph_pool/resnet_small_graph1.pnnx.bin");

  for (int j = 0; j < 3; ++j) {
    graph.Build("pnnx_input_0", "pnnx_output_0");
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 5, 5);
    input->Ones();

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 1);

    const auto &output1 = output_tensors.at(0)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/small_graph_pool/1.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_LE(output1.at(i) - output2.at(i), 1e-6);
    }
  }
}

TEST(test_layer, forward_small_map) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/small_graph/resnet_small_graph1.pnnx.param", "tmp/small_graph/resnet_small_graph1.pnnx.bin");

  for (int i = 0; i < 3; ++i) {
    graph.Build("pnnx_input_0", "pnnx_output_0");
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 5, 5);
    input->Ones();

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 1);

    const auto &output1 = output_tensors.at(0)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/small_graph/1.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(output1.at(j) - output2.at(j), 1e-6);
    }
  }
}

TEST(test_layer, forward_conv1_small) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/im2cols/conv1x1.pnnx.param", "tmp/im2cols/conv1x1.pnnx.bin");
  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(3, 5, 5);
  input_tensor->at(0) = CSVDataLoader::LoadData("./tmp/im2cols/1.csv");
  input_tensor->at(1) = CSVDataLoader::LoadData("./tmp/im2cols/2.csv");
  input_tensor->at(2) = CSVDataLoader::LoadData("./tmp/im2cols/3.csv");
  inputs.push_back(input_tensor);

  graph.Forward(inputs, false);
}

TEST(test_layer, forward_conv1_large) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/im2cols/conv1x1.pnnx.param", "tmp/im2cols/conv1x1.pnnx.bin");
  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(3, 512, 512);
  input_tensor->at(0) = CSVDataLoader::LoadData("./tmp/im2cols/1_large.csv");
  input_tensor->at(1) = CSVDataLoader::LoadData("./tmp/im2cols/2_large.csv");
  input_tensor->at(2) = CSVDataLoader::LoadData("./tmp/im2cols/3_large.csv");
  inputs.push_back(input_tensor);

  const auto &output = graph.Forward(inputs, false);
  ASSERT_EQ(output.size(), 1);
}