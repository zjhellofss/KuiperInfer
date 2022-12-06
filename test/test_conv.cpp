//
// Created by fss on 22-11-22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"

TEST(test_layer, forward_identity_block0) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/resnet_identity/resnet_identity_graph_max2.pnnx.param",
       "tmp/resnet_identity/resnet_identity_graph_max2.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 512, 512);
  input->Ones();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 1);

  const auto &output1 = output_tensors.at(0)->at(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/4.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_LE(abs(output1.at(i) - output2.at(i)), 1e-6);
  }
}

TEST(test_layer, forward_identity_block1) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/resnet_identity/resnet_identity_graph_big.pnnx.param",
       "tmp/resnet_identity/resnet_identity_graph_big.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 512, 512);
  input->Ones();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 1);

  const auto &output1 = output_tensors.at(0)->at(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/1.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  for (uint32_t i = 0; i < size; ++i) {
    ASSERT_LE(abs(output1.at(i) - output2.at(i)), 1e-6);
  }
}

TEST(test_layer, forward_identity_block2) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/resnet_identity/resnet_identity_graph_max.pnnx.param",
       "tmp/resnet_identity/resnet_identity_graph_max.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 512, 512);
  input->Ones();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 1);

  const auto &output1 = output_tensors.at(0)->at(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/2.csv");
  ASSERT_EQ(output1.size(), output2.size());

  const uint32_t size = output1.size();
  double eps_num = 0.;
  for (uint32_t i = 0; i < size; ++i) {
    eps_num += abs(output1.at(i) - output2.at(i));
  }
  ASSERT_LE(eps_num / size, 1e-6);
}

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
      ASSERT_LE(abs(output1.at(i) - output2.at(i)), 1e-6);
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
      ASSERT_LE(abs(output1.at(j) - output2.at(j)), 1e-6);
    }
  }
}

TEST(test_layer, forward_identity_block3) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet_identity/resnet_identity_graph_test.pnnx.param",
                     "tmp/resnet_identity/resnet_identity_graph_test.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  for (int i = 0; i < 3; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 512, 512);
    input->Ones();

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 1);

    const auto &output1 = output_tensors.at(0)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/7.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(abs(output1.at(j) - output2.at(j)), 1e-6);
    }
  }
}

TEST(test_layer, forward_identity_block4) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet_identity/resnet_91_graph.pnnx.param",
                     "tmp/resnet_identity/resnet_91_graph.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  for (int i = 0; i < 3; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 32, 32);
    input1->Fill(1.);

    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 32, 32);
    input2->Fill(1.);

    std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 32, 32);
    input3->Fill(1.);

    std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 32, 32);
    input4->Fill(1.);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    inputs.push_back(input4);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 4);

    const auto &output1 = output_tensors.at(0)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/13.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(abs(output1.at(j) - output2.at(j)), 1e-6);
    }
  }
}

TEST(test_layer, forward_identity_block5) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet_identity/resnet_93_graph.pnnx.param",
                     "tmp/resnet_identity/resnet_93_graph.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  for (int i = 0; i < 3; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 128, 128);
    input1->Fill(1.);

    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 128, 128);
    input2->Fill(1.);

    std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 128, 128);
    input3->Fill(1.);

    std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 128, 128);
    input4->Fill(1.);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    inputs.push_back(input4);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 4);

    const auto &output1 = output_tensors.at(0)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/17.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(abs(output1.at(j) - output2.at(j)), 1e-6);
    }
  }
}

TEST(test_layer, forward_identity_block6) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet_identity/resnet_batchnorm_identity5.pnnx.param",
                     "tmp/resnet_identity/resnet_batchnorm_identity5.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  for (int i = 0; i < 3; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(1.);

    std::shared_ptr<Tensor<float>> input2 = std::make_shared<Tensor<float>>(3, 224, 224);
    input2->Fill(1.);

    std::shared_ptr<Tensor<float>> input3 = std::make_shared<Tensor<float>>(3, 224, 224);
    input3->Fill(1.);

    std::shared_ptr<Tensor<float>> input4 = std::make_shared<Tensor<float>>(3, 224, 224);
    input4->Fill(1.);

    std::shared_ptr<Tensor<float>> input5= std::make_shared<Tensor<float>>(3, 224, 224);
    input5->Fill(1.);

    std::shared_ptr<Tensor<float>> input6 = std::make_shared<Tensor<float>>(3, 224, 224);
    input6->Fill(1.);

    std::shared_ptr<Tensor<float>> input7 = std::make_shared<Tensor<float>>(3, 224, 224);
    input7->Fill(1.);

    std::shared_ptr<Tensor<float>> input8 = std::make_shared<Tensor<float>>(3, 224, 224);
    input8->Fill(1.);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    inputs.push_back(input4);
    inputs.push_back(input5);
    inputs.push_back(input6);
    inputs.push_back(input7);
    inputs.push_back(input8);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 8);

    const auto &output1 = output_tensors.at(0)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/resnet_identity/22.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(abs(output1.at(j) - output2.at(j)), 1e-6);
    }
  }
}
