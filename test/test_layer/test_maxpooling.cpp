//
// Created by fss on 22-11-24.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../../source/layer/details/maxpooling.hpp"

void MaxPooling(const std::vector<std::shared_ptr<kuiper_infer::Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<kuiper_infer::Tensor<float>>> &outputs,
                int stride_w, int stride_h, int kernel_h, int kernel_w) {
  using namespace kuiper_infer;
  const uint32_t batch = inputs.size();
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>> &input_data = inputs.at(i);
    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t output_c = input_c;

    const uint32_t output_h = uint32_t(std::floor((input_h - kernel_h) / stride_h + 1));
    const uint32_t output_w = uint32_t(std::floor((input_w - kernel_w) / stride_w + 1));
    CHECK(output_w > 0 && output_h > 0);

    std::shared_ptr<Tensor<float>> output_data = std::make_shared<Tensor<float>>(output_c, output_h, output_w);
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat &input_channel = input_data->at(ic);
      arma::fmat &output_channel = output_data->at(ic);
      for (uint32_t r = 0; r < input_h - kernel_h + 1; r += stride_h) {
        for (uint32_t c = 0; c < input_w - kernel_w + 1; c += stride_w) {
          const arma::fmat &region = input_channel.submat(r, c, r + kernel_h - 1, c + kernel_w - 1);
          output_channel.at(int(r / stride_h), int(c / stride_w)) = region.max();
        }
      }
    }
    outputs.push_back(output_data);
  }
}

TEST(test_layer, forward_max_pooling_s11_k33) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s11_k33_p1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 226;
  const uint32_t input_w = 226;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s22_k33) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s33_k33) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 3;
  const uint32_t kernel_w = 3;

  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s11_k55) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s22_k55) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s33_k55) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s55_k55) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 5;
  const uint32_t kernel_w = 5;

  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s11_k77) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 1;
  const uint32_t stride_w = 1;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s22_k77) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 2;
  const uint32_t stride_w = 2;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s33_k77) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 3;
  const uint32_t stride_w = 3;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s55_k77) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 5;
  const uint32_t stride_w = 5;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}

TEST(test_layer, forward_max_pooling_s77_k77) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  const uint32_t input_h = 224;
  const uint32_t input_w = 224;
  const uint32_t kernel_h = 7;
  const uint32_t kernel_w = 7;

  const uint32_t stride_h = 7;
  const uint32_t stride_w = 7;
  const uint32_t input_size = 3;

  for (uint32_t i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, input_h, input_w);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  MaxPooling(inputs, outputs1, stride_w, stride_h, kernel_h, kernel_w);
  ASSERT_EQ(outputs1.size(), input_size);
  MaxPoolingLayer max_layer(0, 0, kernel_h, kernel_w, stride_h, stride_w);

  std::vector<std::shared_ptr<Tensor<float>>> outputs2(input_size);
  max_layer.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.size(), input_size);

  for (uint32_t i = 0; i < input_size; ++i) {
    const auto &output1 = outputs1.at(i);
    const auto &output2 = outputs2.at(i);
    ASSERT_EQ(output1->channels(), output2->channels());
    uint32_t channels = output1->channels();
    for (int c = 0; c < channels; ++c) {
      ASSERT_TRUE(arma::approx_equal(output1->at(c), output2->at(c), "absdiff", 0.01f));
    }
  }
}