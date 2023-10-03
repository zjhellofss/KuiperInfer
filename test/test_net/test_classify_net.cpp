// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-22.
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_net, forward_resnet18) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/resnet18_batch1.param", "tmp/resnet/resnet18_batch1.pnnx.bin");
  graph.Build();

  int repeat_number = 2;
  for (int i = 0; i < repeat_number; ++i) {
    std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 224, 224);
    input1->Fill(2.);

    std::vector<std::shared_ptr<Tensor<float>>> inputs;
    inputs.push_back(input1);

    graph.set_inputs("pnnx_input_0", inputs);
    graph.Forward(false);
    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
    ASSERT_EQ(outputs.size(), 1);

    const auto& output2 = CSVDataLoader::LoadData<float>("tmp/resnet/1.csv");
    const auto& output1 = outputs.front()->data().slice(0);
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t s = 0; s < output1.size(); ++s) {
      ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
    }
  }
}

TEST(test_net, forward_group_conv) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/group_conv/group_conv.pnnx.param", "tmp/group_conv/group_conv.pnnx.bin");

  graph.Build();
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(4, 16, 16);
  input1->Fill(2.f);

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
  ASSERT_EQ(outputs.size(), 1);

  const auto& output1 = outputs.front()->data().slice(0);
  const auto& output2 = CSVDataLoader::LoadData<float>("tmp/mobilenet/1.csv");
  ASSERT_EQ(output1.size(), output2.size());
  for (uint32_t s = 0; s < output1.size(); ++s) {
    ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
  }
}

TEST(test_net, forward_mobilenet1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile.pnnx.param", "tmp/mobilenet/mobile.pnnx.bin");

  graph.Build();
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(3, 32, 32);
  input1->Fill(1.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  int repeat_size = 3;
  graph.set_inputs("pnnx_input_0", inputs);

  for (int i = 0; i < repeat_size; ++i) {
    graph.Forward(false);
    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
    ASSERT_EQ(outputs.size(), 1);

    const auto& output1 = outputs.front()->data();
    const auto& output2 = CSVDataLoader::LoadData<float>("tmp/mobilenet/2.csv");
    ASSERT_EQ(output1.size(), output2.size());
    for (uint32_t s = 0; s < output1.size(); ++s) {
      ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
    }
  }
}

TEST(test_net, forward_mobilenet2) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/mobilenet/mobile_224.pnnx.param", "tmp/mobilenet/mobile_224.pnnx.bin");

  graph.Build();
  const uint32_t channels = 3;
  std::shared_ptr<Tensor<float>> input1 = std::make_shared<Tensor<float>>(channels, 224, 224);
  input1->Fill(1.f);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input1);

  graph.set_inputs("pnnx_input_0", inputs);

  int repeat_size = 3;
  for (int i = 0; i < repeat_size; ++i) {
    graph.Forward(false);
    std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
    ASSERT_EQ(outputs.size(), 1);

    const auto& output2 = CSVDataLoader::LoadData<float>("tmp/mobilenet/3.csv");
    const auto& output1 = outputs.front()->data();
    ASSERT_EQ(output1.size(), output2.size());

    for (uint32_t s = 0; s < output1.size(); ++s) {
      ASSERT_LE(std::abs(output1.at(s) - output2.at(s)), 5e-6);
    }
  }
}
