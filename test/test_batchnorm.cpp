//
// Created by fss on 22-11-30.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"

TEST(test_layer, forward_batchnorm1) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/batchnorm/resnet_batchnorm.pnnx.param",
       "tmp/batchnorm/resnet_batchnorm.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 128, 128);
    input1->Fill(1.);
    inputs.push_back(input1);
  }

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);
  const auto &output2 = CSVDataLoader::LoadData("tmp/batchnorm/1.csv");

  for (int i = 0; i < batch_size; ++i) {
    const auto &output1 = output_tensors.at(i)->at(0);
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t j = 0; j < size; ++j) {
      ASSERT_LE(abs(output1.at(j) - output2.at(j)), 1e-6);
    }
  }
}

TEST(test_layer, forward_batchnorm2) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/batchnorm/resnet_batchnorm2.pnnx.param",
       "tmp/batchnorm/resnet_batchnorm2.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 128, 128);
    input1->Fill(1.);
    inputs.push_back(input1);
  }

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);

  for (int j = 0; j < batch_size; ++j) {
    const auto &output1 = output_tensors.at(j)->at(0);
    const auto &output2 = CSVDataLoader::LoadData("tmp/batchnorm/2.csv");
    ASSERT_EQ(output1.size(), output2.size());

    const uint32_t size = output1.size();
    for (uint32_t i = 0; i < size; ++i) {
      ASSERT_LE(abs(output1.at(i) - output2.at(i)), 1e-6);
    }

    const auto &output3 = output_tensors.at(j)->at(63);
    const auto &output4 = CSVDataLoader::LoadData("tmp/batchnorm/3.csv");
    ASSERT_EQ(output3.size(), output4.size());

    const uint32_t size2 = output3.size();
    for (uint32_t i = 0; i < size2; ++i) {
      ASSERT_LE(abs(output3.at(i) - output4.at(i)), 1e-6);
    }
  }
}

TEST(test_layer, forward_batchnorm3) {
  using namespace kuiper_infer;
  RuntimeGraph graph
      ("tmp/batchnorm/resnet_batchnorm_sigmoid.pnnx.param",
       "tmp/batchnorm/resnet_batchnorm_sigmoid.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const int batch_size = 4;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 128, 128);
    input1->Fill(1.);
    inputs.push_back(input1);
  }

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
  ASSERT_EQ(output_tensors.size(), 4);
  for (int j = 0; j < batch_size; ++j) {
    const auto &output3 = output_tensors.at(j)->at(63);
    const auto &output4 = CSVDataLoader::LoadData("tmp/batchnorm/5.csv");
    ASSERT_EQ(output3.size(), output4.size());

    const uint32_t size2 = output3.size();
    for (uint32_t i = 0; i < size2; ++i) {
      ASSERT_LE(abs(output3.at(i) - output4.at(i)), 1e-6);
    }
  }
}