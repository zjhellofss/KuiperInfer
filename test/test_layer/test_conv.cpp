//
// Created by fss on 23-2-6.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/convolution.hpp"
#include "data/tensor.hpp"

using namespace kuiper_infer;

InferStatus Convolution(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs,
    const uint32_t stride_h_, const uint32_t stride_w_,
    const std::vector<sftensor>& weights_) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of convolution layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }
  if (weights_.empty()) {
    LOG(ERROR) << "Weight parameters is empty";
    return InferStatus::kInferFailedWeightParameterError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    if (input->empty()) {
      LOG(ERROR) << "The input feature map of convolution layer is empty";
      return InferStatus::kInferFailedInputEmpty;
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

      uint32_t output_h =
          uint32_t(std::floor((input_h - kernel_h) / stride_h_ + 1));
      uint32_t output_w =
          uint32_t(std::floor((input_w - kernel_w) / stride_w_ + 1));
      if (output_h <= 0 || output_w <= 0) {
        LOG(ERROR) << "The size of the output feature map is less than zero";
        return InferStatus::kInferFailedOutputSizeError;
      }

      if (!output_data) {
        output_data =
            std::make_shared<Tensor<float>>(kernel_count, output_h, output_w);
        output_data->Fill(0.f);
      }

      if (kernel->channels() != input_c) {
        LOG(ERROR) << "The channel of the weight and input is not adapting";
        return InferStatus::kInferFailedChannelParameterError;
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
            output_channel.at(int(r / stride_h_), int(c / stride_w_)) +=
                sum_value;
          }
        }
      }
    }
    CHECK(!output_data->empty());
    outputs.at(i) = output_data;
  }
  return InferStatus::kInferSuccess;
}

TEST(test_layer, convolution3x3x32_stride1x1_padding0) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 8, 8);
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-4);
    }
  }
}

TEST(test_layer, convolution3x3x32_stride1x1_padding2) {
  const uint32_t batch_size = 8;
  std::vector<sftensor> inputs(batch_size);
  std::vector<sftensor> outputs1(batch_size);
  std::vector<sftensor> outputs2(batch_size);

  const uint32_t in_channel = 32;
  for (uint32_t i = 0; i < batch_size; ++i) {
    inputs.at(i) = std::make_shared<ftensor>(in_channel, 226, 226);
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-4);
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
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-4);
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
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;
  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-4);
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
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-3);
#else
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-4);
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
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;
  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
#ifdef _MSC_VER
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-3);
#else
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-4);
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
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 13;
  const uint32_t kernel_w = 13;
  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, true);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-3);
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
    inputs.at(i)->Rand();
  }
  const uint32_t kernel_h = 13;
  const uint32_t kernel_w = 13;
  const uint32_t stride_h = 19;
  const uint32_t stride_w = 19;
  const uint32_t kernel_count = 8;
  std::vector<sftensor> weights;
  for (uint32_t i = 0; i < kernel_count; ++i) {
    sftensor kernel = std::make_shared<ftensor>(in_channel, kernel_h, kernel_w);
    kernel->Rand();
    weights.push_back(kernel);
    sftensor bias = std::make_shared<ftensor>(1, 1, 1);
  }
  Convolution(inputs, outputs1, stride_h, stride_w, weights);
  ConvolutionLayer conv_layer(kernel_count, in_channel, kernel_h, kernel_w, 0,
                              0, stride_h, stride_w, 1, false);
  conv_layer.set_weights(weights);
  conv_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs1.size(), outputs2.size());
  for (uint32_t i = 0; i < outputs1.size(); ++i) {
    ASSERT_EQ(outputs1.at(i)->size(), outputs2.at(i)->size());
    const uint32_t output_size = outputs1.at(i)->size();
    for (uint32_t j = 0; j < output_size; ++j) {
      ASSERT_LE(std::abs(outputs1.at(i)->index(j) - outputs2.at(i)->index(j)),
                1e-3);
    }
  }
}