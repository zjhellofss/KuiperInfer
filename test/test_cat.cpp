//
// Created by fss on 22-12-26.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/cat.hpp"

TEST(test_layer, cat1) {
  using namespace kuiper_infer;
  int input_size = 4;
  int output_size = 2;
  int input_channels = 6;
  int output_channels = 12;

  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  for (int i = 0; i < input_size; ++i) {
    std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_channels, 32, 32);
    input->Rand();
    inputs.push_back(input);
  }
  std::vector<std::shared_ptr<Tensor<float>>> outputs(output_size);
  CatLayer cat_layer(1);
  const auto status = cat_layer.Forward(inputs, outputs);
  // 4 6 --> 2 12
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  for (uint32_t i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->channels(), output_channels);
  }
  ASSERT_TRUE(arma::approx_equal(inputs.at(0)->at(0), outputs.at(0)->at(0), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(0)->at(1), outputs.at(0)->at(1), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(0)->at(2), outputs.at(0)->at(2), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(0)->at(3), outputs.at(0)->at(3), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(0)->at(4), outputs.at(0)->at(4), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(0)->at(5), outputs.at(0)->at(5), "absdiff", 0.01f));

  ASSERT_TRUE(arma::approx_equal(inputs.at(2)->at(0), outputs.at(0)->at(6), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(2)->at(1), outputs.at(0)->at(7), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(2)->at(2), outputs.at(0)->at(8), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(2)->at(3), outputs.at(0)->at(9), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(2)->at(4), outputs.at(0)->at(10), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(2)->at(5), outputs.at(0)->at(11), "absdiff", 0.01f));


  ASSERT_TRUE(arma::approx_equal(inputs.at(1)->at(0), outputs.at(1)->at(0), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(1)->at(1), outputs.at(1)->at(1), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(1)->at(2), outputs.at(1)->at(2), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(1)->at(3), outputs.at(1)->at(3), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(1)->at(4), outputs.at(1)->at(4), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(1)->at(5), outputs.at(1)->at(5), "absdiff", 0.01f));

  ASSERT_TRUE(arma::approx_equal(inputs.at(3)->at(0), outputs.at(1)->at(6), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(3)->at(1), outputs.at(1)->at(7), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(3)->at(2), outputs.at(1)->at(8), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(3)->at(3), outputs.at(1)->at(9), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(3)->at(4), outputs.at(1)->at(10), "absdiff", 0.01f));
  ASSERT_TRUE(arma::approx_equal(inputs.at(3)->at(5), outputs.at(1)->at(11), "absdiff", 0.01f));
}

