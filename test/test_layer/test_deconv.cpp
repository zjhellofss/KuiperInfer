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

// Created by fss on 23-2-6.
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

TEST(test_layer, deconv_nogroup) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/unet/demo_deconv.pnnx.param", "tmp/unet/demo_deconv.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(13, 13, 31);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/unet/test.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}

TEST(test_layer, deconv_group1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/unet/demo_deconv2_.pnnx.param", "tmp/unet/demo_deconv2_.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(12, 13, 31);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/unet/test2.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}

TEST(test_layer, deconv_group2) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/unet/demo_deconv3_.pnnx.param", "tmp/unet/demo_deconv3_.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(16, 16, 31);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/unet/test3.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}

TEST(test_layer, deconv_group_dilation1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/demo_deconv_d_samplept.pnnx.param",
                     "tmp/resnet/demo_deconv_d_samplept.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 2, 2);
    input->at(0, 0, 0) = 1;
    input->at(0, 1, 1) = 1;
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data =
      CSVDataLoader::LoadData<float>("tmp/resnet/test_convtranspose_d_sample.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-5f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i) << " i: " << i;
  }
}

TEST(test_layer, deconv_group_dilation2) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/demo_deconv_dpt11.pnnx.param",
                     "tmp/resnet/demo_deconv_dpt11.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(16, 16, 31);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/resnet/test_convtranspose_d.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}

TEST(test_layer, deconv_group_dilation3) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/demo_deconv_dpt31.pnnx.param",
                     "tmp/resnet/demo_deconv_dpt31.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(16, 16, 31);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/resnet/test_convtranspose_d31.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}