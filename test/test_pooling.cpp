//
// Created by fss on 22-11-24.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/tensor.hpp"
#include "../source/layer/details/maxpooling.hpp"

TEST(test_layer, forward_max_pooling1) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(2, 3, 4);
  arma::fmat data1("1 2 13 31;"
                   "4 6 16 7;"
                   "16 19 21 5");

  arma::fmat data2("1  2   13   31 ;"
                   "4  6   16   7;"
                   "16 19  21   5");

  input_tensor->at(0) = data1;
  input_tensor->at(1) = data2;

  std::vector<std::shared_ptr<Tensor<float>>> input_tensors;
  input_tensors.push_back(input_tensor);
  std::vector<std::shared_ptr<Tensor<float>>> output_tensors;

  std::shared_ptr<Tensor<float>> output_tensor = std::make_shared<Tensor<float>>(2, 2, 3);
  output_tensors.push_back(output_tensor);

  MaxPoolingLayer layer(0, 0, 2, 2, 1, 1);
  const auto &status = layer.Forward(input_tensors, output_tensors);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_EQ(output_tensors.size(), 1);
  ASSERT_EQ(output_tensor->channels(), 2);

  ASSERT_EQ(output_tensor->at(0, 0, 0), 6);
  ASSERT_EQ(output_tensor->at(0, 1, 0), 19);
  ASSERT_EQ(output_tensor->at(0, 0, 2), 31);

  ASSERT_EQ(output_tensor->at(1, 0, 0), 6);
  ASSERT_EQ(output_tensor->at(1, 1, 0), 19);
  ASSERT_EQ(output_tensor->at(1, 0, 2), 31);
  ASSERT_EQ(output_tensor->at(1, 1, 2), 21);

}

TEST(test_layer, forward_max_pooling2) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(2, 3, 4);

  arma::fmat data1("13  22 13 31 ;"
                   "4   56 16 7;"
                   "126 19 21 55");
  input_tensor->at(0) = data1;
  input_tensor->at(1) = data1;

  std::vector<std::shared_ptr<Tensor<float>>> input_tensors;
  input_tensors.push_back(input_tensor);
  std::vector<std::shared_ptr<Tensor<float>>> output_tensors;

  std::shared_ptr<Tensor<float>> output_tensor = std::make_shared<Tensor<float>>(2, 2, 3);
  output_tensors.push_back(output_tensor);

  MaxPoolingLayer layer(0, 0, 2, 2, 1, 1);
  const auto &status = layer.Forward(input_tensors, output_tensors);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  ASSERT_EQ(output_tensors.size(), 1);

  ASSERT_EQ(output_tensor->at(0, 0, 0), 56);
  ASSERT_EQ(output_tensor->at(0, 1, 0), 126);

  ASSERT_EQ(output_tensor->at(1, 0, 0), 56);
  ASSERT_EQ(output_tensor->at(1, 1, 0), 126);
}

TEST(test_layer, max_pooling3) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, 3, 5);
  arma::fmat data1("13  22 13 31 99;"
                   "4   81 56 16 123;"
                   "126 19 21 55 33");
  input_tensor->at(0) = data1;

  std::vector<std::shared_ptr<Tensor<float>>> input_tensors;
  input_tensors.push_back(input_tensor);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors;
  std::shared_ptr<Tensor<float>> output_tensor = std::make_shared<Tensor<float>>(1, 1, 3);
  output_tensors.push_back(output_tensor);

  MaxPoolingLayer layer(0, 0, 3, 3, 2, 1);
  const auto &status = layer.Forward(input_tensors, output_tensors);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(output_tensors.size(), 1);

  ASSERT_EQ(output_tensor->at(0, 0, 0), 126);
  ASSERT_EQ(output_tensor->at(0, 0, 1), 81);
  ASSERT_EQ(output_tensor->at(0, 0, 2), 123);
}

TEST(test_layer, max_pooling4) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, 3, 3);
  arma::fmat data1("13  22 13;"
                   "4   81 56;"
                   "126 19 21");
  input_tensor->at(0) = data1;

  std::vector<std::shared_ptr<Tensor<float>>> input_tensors;
  input_tensors.push_back(input_tensor);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors;
  std::shared_ptr<Tensor<float>> output_tensor = std::make_shared<Tensor<float>>(1, 2, 2);
  output_tensors.push_back(output_tensor);

  MaxPoolingLayer layer(0, 0, 2, 2, 1, 1);
  const auto &status = layer.Forward(input_tensors, output_tensors);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(output_tensors.size(), 1);

  ASSERT_EQ(output_tensor->at(0, 0, 0), 81);
  ASSERT_EQ(output_tensor->at(0, 0, 1), 81);
  ASSERT_EQ(output_tensor->at(0, 1, 0), 126);
  ASSERT_EQ(output_tensor->at(0, 1, 1), 81);
}


TEST(test_layer, max_pooling5) {
  using namespace kuiper_infer;
  std::shared_ptr<Tensor<float>> input_tensor = std::make_shared<Tensor<float>>(1, 3, 3);
  arma::fmat data1("13  22 13;"
                   "4   81 56;"
                   "126 19 21");
  input_tensor->at(0) = data1;

  std::vector<std::shared_ptr<Tensor<float>>> input_tensors;
  input_tensors.push_back(input_tensor);

  std::vector<std::shared_ptr<Tensor<float>>> output_tensors;
  std::shared_ptr<Tensor<float>> output_tensor = std::make_shared<Tensor<float>>(1, 4, 4);
  output_tensors.push_back(output_tensor);

  MaxPoolingLayer layer(1, 1, 2, 2, 1, 1);
  const auto &status = layer.Forward(input_tensors, output_tensors);
  ASSERT_EQ(status, InferStatus::kInferSuccess);
  ASSERT_EQ(output_tensors.size(), 1);

  ASSERT_EQ(output_tensor->at(0, 0, 0), 13);
  ASSERT_EQ(output_tensor->at(0, 0, 1), 22);
  ASSERT_EQ(output_tensor->at(0, 0, 2), 22);
  ASSERT_EQ(output_tensor->at(0, 0, 3), 13);

  ASSERT_EQ(output_tensor->at(0, 2, 0), 126);
  ASSERT_EQ(output_tensor->at(0, 2, 1), 126);
  ASSERT_EQ(output_tensor->at(0, 2, 2), 81);
  ASSERT_EQ(output_tensor->at(0, 2, 3), 56);
}
