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
    
// Created by fss on 22-12-25.
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../../source/layer/details/upsample.hpp"

TEST(test_layer, forward_upsample1) {
  using namespace kuiper_infer;
  UpSampleLayer layer(2.f, 2.f);

  const uint32_t channels = 3;
  const uint32_t rows = 224;
  const uint32_t cols = 224;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->slice(i);
      const auto &input_channel = input->slice(i);
      ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 2);
      ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 2);

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

  const uint32_t channels = 3;
  const uint32_t rows = 224;
  const uint32_t cols = 224;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->slice(i);
      const auto &input_channel = input->slice(i);
      ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 2);
      ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 3);

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
  UpSampleLayer layer(3.f, 2.f);
  const uint32_t channels = 3;
  const uint32_t rows = 224;
  const uint32_t cols = 224;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->slice(i);
      const auto &input_channel = input->slice(i);
      ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 3);
      ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 2);

      for (int r = 0; r < output_channel.n_rows; ++r) {
        for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
          ASSERT_EQ(input_channel.at(r / 3, c_ / 2), output_channel.at(r, c_)) << r << " " << c_;
        }
      }
    }
  }
}

TEST(test_layer, forward_upsample4) {
  using namespace kuiper_infer;
  UpSampleLayer layer(3.f, 3.f);
  const uint32_t channels = 3;
  const uint32_t rows = 224;
  const uint32_t cols = 224;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->slice(i);
      const auto &input_channel = input->slice(i);
      ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 3);
      ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 3);

      for (int r = 0; r < output_channel.n_rows; ++r) {
        for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
          ASSERT_EQ(input_channel.at(r / 3, c_ / 3), output_channel.at(r, c_)) << r << " " << c_;
        }
      }
    }
  }
}

TEST(test_layer, forward_upsample5) {
  using namespace kuiper_infer;
  UpSampleLayer layer(4.f, 4.f);
  const uint32_t channels = 3;
  const uint32_t rows = 224;
  const uint32_t cols = 224;

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(channels, rows, cols);
  input->Rand();

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &output = outputs.at(i);
    for (int c = 0; c < channels; ++c) {
      const auto &output_channel = output->slice(i);
      const auto &input_channel = input->slice(i);
      ASSERT_EQ(output_channel.n_rows / input_channel.n_rows, 4);
      ASSERT_EQ(output_channel.n_cols / input_channel.n_cols, 4);

      for (int r = 0; r < output_channel.n_rows; ++r) {
        for (int c_ = 0; c_ < output_channel.n_cols; ++c_) {
          ASSERT_EQ(input_channel.at(r / 4, c_ / 4), output_channel.at(r, c_)) << r << " " << c_;
        }
      }
    }
  }
}

