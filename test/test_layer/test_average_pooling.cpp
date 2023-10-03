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

// Created by fss on 23-1-3.
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/adaptive_avgpooling.hpp"
#include "data/tensor.hpp"

void AveragePooling(const std::vector<std::shared_ptr<kuiper_infer::Tensor<float>>>& inputs,
                    std::vector<std::shared_ptr<kuiper_infer::Tensor<float>>>& outputs,
                    uint32_t output_h, uint32_t output_w) {
  using namespace kuiper_infer;
  const uint32_t batch = inputs.size();
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t output_c = input_c;

    const uint32_t stride_h = uint32_t(std::floor(input_h / output_h));
    const uint32_t stride_w = uint32_t(std::floor(input_w / output_w));
    const uint32_t kernel_h = input_h - (output_h - 1) * stride_h;
    const uint32_t kernel_w = input_w - (output_w - 1) * stride_w;
    CHECK(output_w > 0 && output_h > 0);

    std::shared_ptr<Tensor<float>> output_data =
        std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_channel = input_data->slice(ic);
      arma::fmat& output_channel = output_data->slice(ic);
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::fmat& region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          output_channel.at(int(r / stride_h), int(c / stride_w)) = arma::mean(arma::mean(region));
        }
      }
    }
    outputs.push_back(output_data);
  }
}

TEST(test_layer, forward_average_pooling_out1x1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 1;
  const uint32_t output_w = 1;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out3x3) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 3;
  const uint32_t output_w = 3;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out3x3p1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 226;
  const uint32_t input_w = 226;
  const uint32_t output_h = 3;
  const uint32_t output_w = 3;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out5x5) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 5;
  const uint32_t output_w = 5;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out7x7) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 7;
  const uint32_t output_w = 7;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out9x9) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 9;
  const uint32_t output_w = 9;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out11x11) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 11;
  const uint32_t output_w = 11;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out1x11) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t output_h = 1;
  const uint32_t output_w = 11;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_average_pooling_out7x11) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 225;
  const uint32_t input_w = 225;
  const uint32_t output_h = 7;
  const uint32_t output_w = 11;

  const uint32_t input_size = 3;
  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->RandN();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  AveragePooling(inputs, outputs1, output_h, output_w);
  ASSERT_EQ(outputs1.size(), input_size);
  AdaptiveAveragePoolingLayer average_layer(output_h, output_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  average_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto& output1 = outputs1.at(i);
    const auto& output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    ASSERT_EQ(output1->shapes(), output2->shapes());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->slice(c), output2->slice(c), "absdiff", 0.01f));
    }
  }
}