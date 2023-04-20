//
// Created by fss on 22-12-23.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/linear.hpp"
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_layer, forward_linear1) {
  using namespace kuiper_infer;
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  LinearLayer linear_layer(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear_layer.set_weights(weights);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features);
    }
  }
}

TEST(test_layer, forward_linear2) {
  using namespace kuiper_infer;
  const uint32_t in_features = 32;
  const uint32_t out_features = 64;
  const uint32_t in_dims = 1280;

  LinearLayer linear_layer(in_features, out_features, false);
  std::vector<float> weights(in_features * out_features, 1.f);
  linear_layer.set_weights(weights);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);
  input->Fill(2.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), in_features * 2);
    }
  }
}

TEST(test_layer, forward_linear3) {
  using namespace kuiper_infer;
  const uint32_t in_features = 8;
  const uint32_t out_features = 12;
  const uint32_t in_dims = 4;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);
  std::vector<float> weights_raw;

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }

  linear_layer.set_weights(weights_raw);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 36);
    }
  }
}

TEST(test_layer, forward_linear4) {
  using namespace kuiper_infer;
  const uint32_t in_features = 64;
  const uint32_t out_features = 128;
  const uint32_t in_dims = 4;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);
  std::vector<float> weights_raw;

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }

  linear_layer.set_weights(weights_raw);
  input->Fill(1.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 2080 * 1.f);
    }
  }
}

TEST(test_layer, forward_linear5) {
  using namespace kuiper_infer;
  const uint32_t in_features = 64;
  const uint32_t out_features = 128;
  const uint32_t in_dims = 4;
  LinearLayer linear_layer(in_features, out_features, false);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(1, in_dims, in_features);
  std::vector<float> weights_raw;

  for (int i = 0; i < out_features; ++i) {
    for (int j = 0; j < in_features; ++j) {
      weights_raw.push_back(float(j + 1));
    }
  }

  linear_layer.set_weights(weights_raw);
  input->Fill(2.f);

  std::shared_ptr<Tensor<float>> output =
      std::make_shared<Tensor<float>>(1, in_dims, out_features);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  outputs.push_back(output);

  const auto status = linear_layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    const auto& output_tensor = outputs.at(i);
    for (int j = 0; j < output_tensor->size(); ++j) {
      ASSERT_EQ(output_tensor->index(j), 2080 * 2.f);
    }
  }
}

TEST(test_layer, forward_linear7) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/linear/linear_512_1000.pnnx.param",
                     "tmp/linear/linear_512_1000.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<sftensor> inputs;

  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(1, 32, 512);
    input->Fill(1.f);
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs;
  outputs = graph.Forward(inputs);
  arma::fmat real_data =
      CSVDataLoader::LoadData("tmp/linear/linear_512x1000.csv");
  for (const auto& output : outputs) {
    for (uint32_t c = 0; c < output->channels(); ++c) {
      bool is_same =
          arma::approx_equal(real_data, output->slice(c), "absdiff", 1e-4f);
      ASSERT_EQ(is_same, true);
    }
  }
}

TEST(test_layer, forward_linear8) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/linear/linear_1305_2407.pnnx.param",
                     "tmp/linear/linear_1305_2407.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<sftensor> inputs;

  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(1, 31, 1305);
    input->Fill(1.f);
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs;
  outputs = graph.Forward(inputs);
  arma::fmat real_data =
      CSVDataLoader::LoadData("tmp/linear/linear_1305x2047.csv");
  for (const auto& output : outputs) {
    for (uint32_t c = 0; c < output->channels(); ++c) {
      bool is_same =
          arma::approx_equal(real_data, output->slice(c), "absdiff", 1e-4f);
      ASSERT_EQ(is_same, true);
    }
  }
}

TEST(test_layer, forward_linear9) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/linear/linear_1305_2407.pnnx.param",
                     "tmp/linear/linear_1305_2407.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<sftensor> inputs;

  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(1, 31, 1305);
    input->Fill(1.f);
    inputs.push_back(input);
  }
  std::vector<sftensor> outputs;
  outputs = graph.Forward(inputs);
  arma::fmat real_data =
      CSVDataLoader::LoadData("tmp/linear/linear_1305x2047.csv");
  for (const auto& output : outputs) {
    for (uint32_t c = 0; c < output->channels(); ++c) {
      bool is_same =
          arma::approx_equal(real_data, output->slice(c), "absdiff", 1e-4f);
      ASSERT_EQ(is_same, true);
    }
  }
}

TEST(test_layer, forward_linear10) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/linear/linear_1305_2407.pnnx.param",
                     "tmp/linear/linear_1305_2407.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
  const uint32_t batch_size = 4;
  std::vector<sftensor> inputs;
  std::vector<float> values;
  for (int i = 0; i < 31 * 1305; ++i) {
    values.push_back(float(i));
  }
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(1, 31, 1305);
    input->Fill(values);
    inputs.push_back(input);
  }

  std::vector<sftensor> outputs;
  outputs = graph.Forward(inputs);
  outputs.front()->Show();
  arma::fmat real_data =
      CSVDataLoader::LoadData("tmp/linear/linear_1305x2047_arange.csv");
  for (const auto& output : outputs) {
    for (uint32_t c = 0; c < output->channels(); ++c) {
      bool is_same =
          arma::approx_equal(real_data, output->slice(c), "absdiff", 1e-1f);
      ASSERT_EQ(is_same, true);
    }
  }
}