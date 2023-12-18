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
#include "../../source/layer/details/convolution.hpp"
#include "data/load_data.hpp"
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

using namespace kuiper_infer;

StatusCode Convolution(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                       std::vector<std::shared_ptr<Tensor<float>>>& outputs,
                       const uint32_t stride_h_, const uint32_t stride_w_,
                       const std::vector<sftensor>& weights_) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of convolution layer is empty";
    return StatusCode::kInferInputsEmpty;
  }
  if (weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return StatusCode::kInferParameterError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of convolution layer is empty";
      return StatusCode::kInferInputsEmpty;
    }
    const uint32_t input_w = input->cols();
    const uint32_t input_h = input->rows();
    const uint32_t input_c = input->channels();

    const uint32_t kernel_count = weights_.size();
    std::shared_ptr<Tensor<float>> output_data;

    for (uint32_t k = 0; k < kernel_count; ++k) {
      const std::shared_ptr<Tensor<float>>& kernel = weights_.at(k);
      const uint32_t kernel_h = kernel->rows();
      const uint32_t kernel_w = kernel->cols();

      uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
      uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));
      if (output_h <= 0 || output_w <= 0) {
        LOG(ERROR) << "The size of the output feature map is less than zero";
        return StatusCode::kInferParameterError;
      }

      if (!output_data) {
        output_data = std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
        output_data->Fill(0.f);
      }

      if (kernel->channels() != input_c) {
        LOG(ERROR) << "The channel of the weight and input is not adapting";
        return StatusCode::kInferParameterError;
      }
      arma::fmat& output_channel = output_data->slice(k);
      for (uint32_t ic = 0; ic < input_c; ++ic) {
        const arma::fmat& input_channel = input->slice(ic);
        const arma::fmat& kernel_channel = kernel->slice(ic);

        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w_) {
          for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h_) {
            const arma::fmat& region =
                input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
            const float sum_value = arma::accu(region % kernel_channel);
            output_channel.at(int(r / stride_h_), int(c / stride_w_)) += sum_value;
          }
        }
      }
    }
    CHECK(!output_data->empty());
    outputs.at(i) = output_data;
  }
  return StatusCode::kSuccess;
}

// TEST(test_layer, convolution3x3_winograd1) {
//   sftensor input = std::make_shared<ftensor>(3, 5, 5);
//   input->Fill(1.f);
//   sftensor weight1 = std::make_shared<ftensor>(3, 3, 3);
//   weight1->RandN();

//   sftensor weight2 = std::make_shared<ftensor>(3, 3, 3);
//   weight2->RandN();

//   std::vector<sftensor> weights(2);
//   weights.at(0) = weight1;
//   weights.at(1) = weight2;

//   sftensor output;
//   Convolution3x3s1(input, output, weights);

//   std::vector<sftensor> inputs;
//   inputs.push_back(input);
//   std::vector<sftensor> outputs(1);
//   Convolution(inputs, outputs, 1, 1, weights);
//   ASSERT_EQ(outputs.size(), 1);
//   ASSERT_EQ(TensorIsSame(outputs.front(), output, 1e-4), true);
// }

// TEST(test_layer, convolution3x3_winograd2) {
//   sftensor input = std::make_shared<ftensor>(31, 51, 51);
//   input->Fill(1.f);

//   const uint32_t kernel_count = 8;
//   std::vector<sftensor> weights(kernel_count);
//   for (uint32_t k = 0; k < kernel_count; ++k) {
//     sftensor weight = std::make_shared<ftensor>(31, 3, 3);
//     weight->RandN();
//     weights.at(k) = weight;
//   }

//   sftensor output;
//   Convolution3x3s1(input, output, weights);

//   std::vector<sftensor> inputs;
//   inputs.push_back(input);
//   std::vector<sftensor> outputs(1);
//   Convolution(inputs, outputs, 1, 1, weights);
//   ASSERT_EQ(outputs.size(), 1);
//   ASSERT_EQ(TensorIsSame(outputs.front(), output, 1e-1), true);
// }

// TEST(test_layer, convolution3x3_winograd3) {
//   sftensor input = std::make_shared<ftensor>(131, 511, 511);
//   input->Fill(1.f);

//   const uint32_t kernel_count = 8;
//   std::vector<sftensor> weights(kernel_count);
//   for (uint32_t k = 0; k < kernel_count; ++k) {
//     sftensor weight = std::make_shared<ftensor>(131, 3, 3);
//     weight->RandN();
//     weights.at(k) = weight;
//   }

//   sftensor output;
//   Convolution3x3s1(input, output, weights);

//   std::vector<sftensor> outputs(1);
//   std::vector<sftensor> inputs;
//   inputs.push_back(input);
//   ConvolutionLayer conv_layer(kernel_count, 131, 3, 3, 0,
//   0,
//                               1, 1, 1, false);
//   conv_layer.set_weights(weights);
//   conv_layer.Forward(inputs, outputs);
//   ASSERT_EQ(outputs.size(), 1);
//   ASSERT_EQ(TensorIsSame(outputs.front(), output, 5e-4), true);
// }

// TEST(test_layer, convolution3x3_winograd4) {
//   sftensor input = std::make_shared<ftensor>(13, 1011, 511);
//   input->Fill(1.f);

//   const uint32_t kernel_count = 8;
//   std::vector<sftensor> weights(kernel_count);
//   for (uint32_t k = 0; k < kernel_count; ++k) {
//     sftensor weight = std::make_shared<ftensor>(13, 3, 3);
//     weight->RandN();
//     weights.at(k) = weight;
//   }

//   sftensor output;
//   Convolution3x3s1(input, output, weights);

//   std::vector<sftensor> outputs(1);
//   std::vector<sftensor> inputs;
//   inputs.push_back(input);
//   ConvolutionLayer conv_layer(kernel_count, 13, 3, 3, 0, 0,
//   1,
//                               1, 1, false);
//   conv_layer.set_weights(weights);
//   conv_layer.Forward(inputs, outputs);
//   ASSERT_EQ(outputs.size(), 1);
//   ASSERT_EQ(TensorIsSame(outputs.front(), output, 5e-4), true);
// }

// TEST(test_layer, convolution3x3_winograd5) {
//   sftensor input = std::make_shared<ftensor>(256, 16, 16);
//   input->RandN();

//   const uint32_t kernel_count = 8;
//   std::vector<sftensor> weights(kernel_count);
//   for (uint32_t k = 0; k < kernel_count; ++k) {
//     sftensor weight = std::make_shared<ftensor>(256, 3, 3);
//     weight->RandN();
//     weights.at(k) = weight;
//   }

//   sftensor output;
//   Convolution3x3s1(input, output, weights);

//   std::vector<sftensor> outputs(1);
//   std::vector<sftensor> inputs;
//   inputs.push_back(input);
//   ConvolutionLayer conv_layer(kernel_count, 256, 3, 3, 0,
//   0,
//                               1, 1, 1, false);
//   conv_layer.set_weights(weights);
//   conv_layer.Forward(inputs, outputs);
//   ASSERT_EQ(outputs.size(), 1);
//   ASSERT_EQ(TensorIsSame(outputs.front(), output, 5e-4), true);
// }

TEST(test_layer, convolution3x3x32_stride1x1_padding0) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 8, 8);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-4);
    }
  }
}

TEST(test_layer, convolution1x1x3_stride1x1_padding0) {
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 5;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 151, 233);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 1;
  const uint32_t kernel_w = 1;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 37;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-4);
    }
  }
}

TEST(test_layer, convolution3x5x32_stride1x3_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 3;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 5e-4);
    }
  }
}

TEST(test_layer, convolution3x3x32_stride2x2_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-4);
    }
  }
}

TEST(test_layer, convolution3x3x32_stride5x5_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-4);
    }
  }
}

TEST(test_layer, convolution5x5x32_stride5x5_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-3);
#else
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-4);
#endif
    }
  }
}

TEST(test_layer, convolution5x5x32_stride7x7_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-3);
#else
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-4);
#endif
    }
  }
}

TEST(test_layer, convolution13x13x32_stride7x7_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 13;
  const uint32_t kernel_w = 13;
  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, true);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-2);
#else
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-3);
#endif
    }
  }
}

TEST(test_layer, convolution13x13x31_stride19x19_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 31;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 340, 340);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 13;
  const uint32_t kernel_w = 13;
  const uint32_t stride_h = 19;
  const uint32_t stride_w = 19;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-3);
    }
  }
}

#ifndef _MSC_VER
TEST(test_layer, convolution1x1x1_stride1x1_padding0) {
  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 31;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 41, 4);
    inputs.at(i)->RandN();
  }
  const uint32_t kernel_h = 1;
  const uint32_t kernel_w = 1;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 31;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->RandN();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0, 0, stride_h,
                              stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)), 1e-3);
    }
  }
}
#endif

TEST(test_layer, conv3x3_fromtorch) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/conv1.pnnx.param", "tmp/resnet/conv1.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(18, 5, 5);
    std::vector<float> values;
    for (int j = 0; j < 450; ++j) {
      values.push_back((float)j);
    }
    input->Fill(values, true);
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  const std::vector<float> outputs_values = outputs.front()->values(true);
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/resnet/test13.csv");
  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 1e-4f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}

TEST(test_layer, conv_dilation1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/dilation_conv_pt.pnnx.param",
                     "tmp/resnet/dilation_conv_pt.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 25, 25);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/resnet/test_dilation.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}

TEST(test_layer, conv_dilation2) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/resnet/dilation_conv_pt2.pnnx.param",
                     "tmp/resnet/dilation_conv_pt2.pnnx.bin");

  graph.Build();
  const uint32_t batch_size = 1;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;

  for (int i = 0; i < batch_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 36, 36);
    input->Ones();
    inputs.push_back(input);
  }

  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);

  std::vector<sftensor> outputs = graph.get_outputs("pnnx_output_0");
  arma::fmat real_data = CSVDataLoader::LoadData<float>("tmp/resnet/test_dilation2.csv");
  const auto& outputs_values = outputs.front()->values(true);

  for (int i = 0; i < outputs_values.size(); ++i) {
    ASSERT_LE(std::abs(real_data.at(i) - outputs_values.at(i)), 2e-6f)
        << i << " real: " << real_data.at(i) << " predict: " << outputs_values.at(i);
  }
}
