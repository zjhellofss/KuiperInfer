//
// Created by fss on 22-11-12.
//
#include <gtest/gtest.h>
#include "../source/layer/flatten.hpp"
#include "../source/layer/avgpooling.hpp"
#include "../source/layer/sigmoid.hpp"
#include "../source/layer/linear.hpp"

TEST(test_layer, linear1) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 4;
  const uint32_t cols = 1;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, cols, rows);
  input_data->Fill(1.);
  std::vector<std::shared_ptr<Blob>> input_datas;
  input_datas.push_back(input_data);

  std::shared_ptr<Blob> weight_data = std::make_shared<Blob>(channels, rows, cols);
  std::shared_ptr<Blob> bias_data = std::make_shared<Blob>(channels, cols, cols);

  weight_data->Fill(1.);
  bias_data->Fill(2.);

  std::vector<std::shared_ptr<Blob>> weight_datas;
  weight_datas.push_back(weight_data);

  std::vector<std::shared_ptr<Blob>> bias_datas;
  bias_datas.push_back(bias_data);

  std::vector<std::shared_ptr<Blob>> output_datas;
  LinearLayer layer(weight_datas, bias_datas);
  layer.Forward(input_datas, output_datas);

  ASSERT_EQ(output_datas.size(), input_datas.size());
  const uint32_t batch_size = output_datas.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    std::shared_ptr<Blob> output_data = output_datas.at(i);
    ASSERT_EQ(output_data->rows(), 1);
    ASSERT_EQ(output_data->cols(), 1);
    ASSERT_EQ(output_data->channels(), 3);
    for (uint32_t c = 0; c < channels; ++c) {
      const auto &mat = output_data->at(c);
      ASSERT_EQ(mat.at(0, 0), 6);
    }
  }
}

TEST(test_layer, linear2) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 4;
  const uint32_t cols = 1;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, cols, rows);
  input_data->Fill(1.);
  std::vector<std::shared_ptr<Blob>> input_datas;
  input_datas.push_back(input_data);

  std::shared_ptr<Blob> weight_data = std::make_shared<Blob>(channels, rows, cols);
  std::shared_ptr<Blob> bias_data = std::make_shared<Blob>(channels, cols, cols);

  std::vector<std::shared_ptr<Blob>> weight_datas;
  weight_datas.push_back(weight_data);

  std::vector<std::shared_ptr<Blob>> bias_datas;
  bias_datas.push_back(bias_data);

  std::vector<std::shared_ptr<Blob>> output_datas;

  std::vector<double> weight_datas_(channels * rows * cols, 1.);
  std::vector<double> bias_datas_(channels * cols * cols, 2.);


  LinearLayer layer(weight_datas, bias_datas);
  layer.set_weights(weight_datas_);
  layer.set_bias(bias_datas_);
  layer.Forward(input_datas, output_datas);

  ASSERT_EQ(output_datas.size(), input_datas.size());
  const uint32_t batch_size = output_datas.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    std::shared_ptr<Blob> output_data = output_datas.at(i);
    ASSERT_EQ(output_data->rows(), 1);
    ASSERT_EQ(output_data->cols(), 1);
    ASSERT_EQ(output_data->channels(), 3);
    for (uint32_t c = 0; c < channels; ++c) {
      const auto &mat = output_data->at(c);
      ASSERT_EQ(mat.at(0, 0), 6);
    }
  }
}

TEST(test_layer, sigmoid) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 32;
  const uint32_t cols = 64;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, rows, cols);
  input_data->Fill(0.);

  std::vector<std::shared_ptr<Blob>> input_datas;
  input_datas.push_back(input_data);
  std::vector<std::shared_ptr<Blob>> output_datas;

  SigmoidLayer layer;
  layer.Forward(input_datas, output_datas);
  ASSERT_EQ(input_datas.size(), output_datas.size());

  for (int i = 0; i < input_datas.size(); ++i) {
    const std::shared_ptr<Blob> &output_data = output_datas.at(i);

    for (uint32_t c = 0; c < output_data->channels(); ++c) {
      for (uint32_t r = 0; r < output_data->rows(); ++r) {
        for (uint32_t c_ = 0; c_ < output_data->cols(); ++c_) {
          ASSERT_EQ(output_data->at(c, r, c_), 0.5);
        }
      }
    }
  }
}

TEST(test_layer, flatten) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 32;
  const uint32_t cols = 64;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, rows, cols);
  input_data->Fill(2.);
  std::vector<std::shared_ptr<Blob>> input_datas;
  std::vector<std::shared_ptr<Blob>> output_datas;

  input_datas.push_back(input_data);

  FlattenLayer layer;
  layer.Forward(input_datas, output_datas);
  ASSERT_EQ(output_datas.size(), input_datas.size());

  for (int i = 0; i < input_datas.size(); ++i) {
    const std::shared_ptr<Blob> &output_data = output_datas.at(i);
    const std::shared_ptr<Blob> &input_data_ = input_datas.at(i);

    ASSERT_EQ(output_data->channels(), 1);
    ASSERT_EQ(output_data->rows(), 1);
    ASSERT_EQ(output_data->cols(), rows * cols * channels);
    ASSERT_EQ(output_data->front(), 2.);
    ASSERT_EQ(input_data_->channels(), 3);
    ASSERT_EQ(input_data_->rows(), rows);
  }
}

TEST(test_layer, avgpooling) {
  using namespace kuiper_infer;
  AveragePoolingLayer layer(2, 0, 1);
  std::vector<std::shared_ptr<Blob>> input_datas;
  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(3, 3, 3);
  input_data->Fill(2.);
  input_datas.push_back(input_data);

  std::vector<std::shared_ptr<Blob>> output_datas;
  layer.Forward(input_datas, output_datas);

  ASSERT_EQ(input_datas.size(), output_datas.size());

  uint32_t size = output_datas.size();
  for (uint32_t i = 0; i < size; ++i) {
    const std::shared_ptr<Blob> output_data = output_datas.at(i);
    ASSERT_EQ(output_data->channels(), 3);
    ASSERT_EQ(output_data->rows(), 2);
    ASSERT_EQ(output_data->cols(), 2);
    for (uint32_t c = 0; c < output_data->channels(); ++c) {
      for (uint32_t r = 0; r < output_data->rows(); ++r) {
        for (uint32_t c_ = 0; c_ < output_data->cols(); ++c_) {
          ASSERT_EQ(output_data->at(c, r, c_), 2);
        }
      }
    }
  }
}