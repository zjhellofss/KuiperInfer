//
// Created by fss on 22-11-30.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "data/load_data.hpp"
#include "../source/layer/details/batchnorm2d.hpp"

TEST(test_layer, forward_batchnorm1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
  input->Rand();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(3, 1e-5f, {1, 1, 1}, {0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (const auto &output : outputs) {
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
  input->Rand();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(1, 1e-5f, {1}, {0});
  layer.set_weights(std::vector<float>{0});
  layer.set_bias(std::vector<float>{1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (const auto &output : outputs) {
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
  input->Rand();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(1, 1e-5f, {1}, {0});
  layer.set_weights(std::vector<float>{0});
  layer.set_bias(std::vector<float>{1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (const auto &output : outputs) {
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
  input->Rand();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(5, 1e-5f, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (const auto &output : outputs) {
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
  input->Rand();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(7, 1e-5f, {1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0, 0, 0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1, 1, 1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (const auto &output : outputs) {
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

  RuntimeGraph graph("tmp/batchnorm/bn1.pnnx.param",
                     "tmp/batchnorm/bn1.pnnx.bin");
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs);
  ASSERT_EQ(outputs.size(), 8);
  const auto &output_ = outputs.at(0);
  ASSERT_EQ(output_->channels(), 32);
  for (int i = 0; i < 32; ++i) {
    const std::string &path = "tmp/batchnorm/bn_" + std::to_string(i) + ".csv";
    const auto &output_data1 = CSVDataLoader::LoadData(path);
    const auto &output_data2 = output_->at(i);
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

  RuntimeGraph graph("tmp/batchnorm/bn2.pnnx.param",
                     "tmp/batchnorm/bn2.pnnx.bin");
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.Forward(inputs);
  ASSERT_EQ(outputs.size(), 8);
  const auto &output_ = outputs.at(0);
  ASSERT_EQ(output_->channels(), 32);
  for (int i = 0; i < 32; ++i) {
    const std::string &path = "tmp/batchnorm/bn2_" + std::to_string(i) + ".csv";
    const auto &output_data1 = CSVDataLoader::LoadData(path);
    const auto &output_data2 = output_->at(i);
    ASSERT_TRUE(arma::approx_equal(output_data1, output_data2, "absdiff", 1e-4));
  }
}