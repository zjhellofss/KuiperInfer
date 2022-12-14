//
// Created by fss on 22-11-22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"

TEST(test_layer, forward_resnet18) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/resnet18_batch1.param",
                     "tmp/resnet/resnet18_batch1.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  int repeat_number = 3;
  for (int i = 0; i < repeat_number; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    std::vector<std::shared_ptr<Tensor<float>>> output_tensors = graph.Forward(inputs, false);
    ASSERT_EQ(output_tensors.size(), 1);

    const auto &output1 = output_tensors.at(0)->data();
    const auto &output2 = CSVDataLoader::LoadData("tmp/resnet/23.csv");

    uint32_t size1 = output1.size();
    uint32_t size2 = output2.size();
    ASSERT_EQ(size1, size2);
    for (uint32_t s = 0; s < size1; ++s) {
      ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
    }
  }
}

TEST(test_layer, forward_group_conv) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/group_conv/group_conv.pnnx.param",
                     "tmp/group_conv/group_conv.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(4, 16, 16);
  input1->Fill(2.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);

  const auto &output1 = outputs.front()->data().slice(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/mobilenet/1.csv");
  ASSERT_EQ(output1.size(), output2.size());
  for (uint32_t s = 0; s < output1.size(); ++s) {
    ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
  }
}

TEST(test_layer, forward_mobilenet) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile.pnnx.param",
                     "tmp/mobilenet/mobile.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 32, 32);
  input1->Fill(1.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);

  const auto &output1 = outputs.front()->data().slice(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/mobilenet/2.csv");
  ASSERT_EQ(output1.size(), output2.size());
  for (uint32_t s = 0; s < output1.size(); ++s) {
    ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
  }
}

TEST(test_layer, forward_mobilenet_224) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile_224.pnnx.param",
                     "tmp/mobilenet/mobile_224.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
  input1->Fill(1.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs, false);
  ASSERT_EQ(outputs.size(), 1);

  const auto &output1 = outputs.front()->data().slice(0);
  const auto &output2 = CSVDataLoader::LoadData("tmp/mobilenet/3.csv");
  ASSERT_EQ(output1.size(), output2.size());
  for (uint32_t s = 0; s < output1.size(); ++s) {
    ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
  }
}
