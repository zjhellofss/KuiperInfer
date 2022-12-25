//
// Created by fss on 22-12-25.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../source/layer/details/upsample.hpp"

TEST(test_layer, forward_upsample1) {
  using namespace kuiper_infer;
  UpSampleLayer layer(2.f, 2.f);
  std::vector<float> values;
  const uint32_t elem_sizes = 48;
  for (int i = 0; i < elem_sizes; ++i) {
    values.push_back(float(i));
  }
  const uint32_t channels = 2;
  const uint32_t rows = 6;
  const uint32_t cols = 4;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  for (int i = 0; i < 4; ++i) {
    input->Fill(values);
  }
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->at(i);
      const auto &input_channel = input->at(i);
      for (int r = 0; r < output_channel.n_rows; ++r) {
        for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
          ASSERT_EQ(input_channel.at(r / 2, c_ / 2), output_channel.at(r, c_)) << r << " " << c_;
        }
      }
    }
  }
}

TEST(test_layer, forward_upsample2) {
  using namespace kuiper_infer;
  UpSampleLayer layer(2.f, 3.f);
  std::vector<float> values;
  const uint32_t elem_sizes = 48;
  for (int i = 0; i < elem_sizes; ++i) {
    values.push_back(float(i));
  }
  const uint32_t channels = 2;
  const uint32_t rows = 6;
  const uint32_t cols = 4;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  for (int i = 0; i < 4; ++i) {
    input->Fill(values);
  }
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->at(i);
      const auto &input_channel = input->at(i);
      for (int r = 0; r < output_channel.n_rows; ++r) {
        for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
          ASSERT_EQ(input_channel.at(r / 2, c_ / 3), output_channel.at(r, c_)) << r << " " << c_;
        }
      }
    }
  }
}


TEST(test_layer, forward_upsample3) {
  using namespace kuiper_infer;
  UpSampleLayer layer(4.f, 4.f);
  std::vector<float> values;
  const uint32_t elem_sizes = 48;
  for (int i = 0; i < elem_sizes; ++i) {
    values.push_back(float(i));
  }
  const uint32_t channels = 2;
  const uint32_t rows = 6;
  const uint32_t cols = 4;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  for (int i = 0; i < 4; ++i) {
    input->Fill(values);
  }
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->at(i);
      const auto &input_channel = input->at(i);
      for (int r = 0; r < output_channel.n_rows; ++r) {
        for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
          ASSERT_EQ(input_channel.at(r / 4, c_ /4), output_channel.at(r, c_)) << r << " " << c_;
        }
      }
    }
  }
}


