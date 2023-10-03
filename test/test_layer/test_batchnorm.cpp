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

// Created by fss on 22-11-30.
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/batchnorm2d.hpp"
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_layer, forward_batchnorm1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
  input->RandN();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(3, 1e-5f, {1, 1, 1}, {0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  for (const auto& output : outputs) {
    const uint32_t size = output->size();
    float mean = 0.f;
    float var = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      mean += output->index(i);
      var += output->index(i) * output->index(i);
    }
    ASSERT_NEAR(mean / size, 0.f, 0.01f);
    ASSERT_NEAR(var / size, 1.f, 0.01f);
  }
}

TEST(test_layer, forward_batchnorm2) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 256 * 512, 1);
  input->RandN();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(1, 1e-5f, {1}, {0});
  layer.set_weights(std::vector<float>{0});
  layer.set_bias(std::vector<float>{1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  for (const auto& output : outputs) {
    const uint32_t size = output->size();
    float mean = 0.f;
    float var = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      mean += output->index(i);
      var += output->index(i) * output->index(i);
    }
    ASSERT_NEAR(mean / size, 0.f, 0.01f);
    ASSERT_NEAR(var / size, 1.f, 0.01f);
  }
}

TEST(test_layer, forward_batchnorm3) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 256, 512);
  input->RandN();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(1, 1e-5f, {1}, {0});
  layer.set_weights(std::vector<float>{0});
  layer.set_bias(std::vector<float>{1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  for (const auto& output : outputs) {
    const uint32_t size = output->size();
    float mean = 0.f;
    float var = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      mean += output->index(i);
      var += output->index(i) * output->index(i);
    }
    ASSERT_NEAR(mean / size, 0.f, 0.01f);
    ASSERT_NEAR(var / size, 1.f, 0.01f);
  }
}

TEST(test_layer, forward_batchnorm4) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(5, 256, 512);
  input->RandN();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(5, 1e-5f, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  for (const auto& output : outputs) {
    const uint32_t size = output->size();
    float mean = 0.f;
    float var = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      mean += output->index(i);
      var += output->index(i) * output->index(i);
    }
    ASSERT_NEAR(mean / size, 0.f, 0.01f);
    ASSERT_NEAR(var / size, 1.f, 0.01f);
  }
}

TEST(test_layer, forward_batchnorm5) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(7, 512, 512);
  input->RandN();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(7, 1e-5f, {1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0, 0, 0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1, 1, 1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, StatusCode::kSuccess);

  for (const auto& output : outputs) {
    const uint32_t size = output->size();
    float mean = 0.f;
    float var = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      mean += output->index(i);
      var += output->index(i) * output->index(i);
    }
    ASSERT_NEAR(mean / size, 0.f, 0.01f);
    ASSERT_NEAR(var / size, 1.f, 0.01f);
  }
}

TEST(test_layer, forward_batchnorm6) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  int batch_size = 8;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 16, 16);
    input->Ones();
    inputs.push_back(input);
  }

  RuntimeGraph graph("tmp/batchnorm/bn1.pnnx.param", "tmp/batchnorm/bn1.pnnx.bin");
  graph.Build();
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
  ASSERT_EQ(outputs.size(), 8);
  const auto& output_ = outputs.at(0);
  ASSERT_EQ(output_->channels(), 32);
  for (int i = 0; i < 32; ++i) {
    const std::string& path = "tmp/batchnorm/bn_" + std::to_string(i) + ".csv";
    const auto& output_data1 = CSVDataLoader::LoadData<float>(path);
    const auto& output_data2 = output_->slice(i);
    ASSERT_TRUE(arma::approx_equal(output_data1, output_data2, "absdiff", 1e-6));
  }
}

TEST(test_layer, forward_batchnorm7) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  int batch_size = 8;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 16, 16);
    input->Ones();
    inputs.push_back(input);
  }

  RuntimeGraph graph("tmp/batchnorm/bn2.pnnx.param", "tmp/batchnorm/bn2.pnnx.bin");
  graph.Build();
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");
  ASSERT_EQ(outputs.size(), 8);
  const auto& output_ = outputs.at(0);
  ASSERT_EQ(output_->channels(), 32);
  for (int i = 0; i < 32; ++i) {
    const std::string& path = "tmp/batchnorm/bn2_" + std::to_string(i) + ".csv";
    const auto& output_data1 = CSVDataLoader::LoadData<float>(path);
    const auto& output_data2 = output_->slice(i);
    ASSERT_TRUE(arma::approx_equal(output_data1, output_data2, "absdiff", 1e-4));
  }
}