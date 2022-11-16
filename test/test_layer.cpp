//
// Created by fss on 22-11-12.
//
#include <gtest/gtest.h>
#include "../source/layer/flatten.hpp"
#include "../source/layer/avgpooling.hpp"
#include "../source/layer/sigmoid.hpp"
#include "../source/layer/linear.hpp"
#include "../source/layer/softmax.hpp"
#include "../source/layer/convolution.hpp"

TEST(test_layer, conv1) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 32;
  const uint32_t cols = 32;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, cols, rows);
  input_data->Fill(1.);

  const uint32_t batch = 16;
  std::vector<std::shared_ptr<Blob>> input_datas;
  for (uint32_t b = 0; b < batch; ++b) {
    input_datas.push_back(input_data);
  }

  const uint32_t kernel_rows = 3;
  const uint32_t kernel_cols = 3;
  std::shared_ptr<Blob> weight_data = std::make_shared<Blob>(channels, kernel_rows, kernel_cols);
  std::shared_ptr<Blob> bias_data = std::make_shared<Blob>(channels, kernel_rows, kernel_cols);

  weight_data->Fill(1.);
  bias_data->Fill(2.);

  const uint32_t output_channel = 8;
  std::vector<std::shared_ptr<Blob>> weight_datas;
  std::vector<std::shared_ptr<Blob>> bias_datas;
  for (uint32_t b = 0; b < output_channel; ++b) {
    weight_datas.push_back(weight_data);
    bias_datas.push_back(bias_data);
  }

  std::vector<std::shared_ptr<Blob>> output_datas;
  ConvolutionLayer layer(weight_datas, bias_datas, 0, 1);
  layer.Forward(input_datas, output_datas);

  ASSERT_EQ(output_datas.size(), batch);
  for (int i = 0; i < output_channel; ++i) {
    const auto output_data = output_datas.at(i);
    ASSERT_EQ(output_data->channels(), output_channel);
    ASSERT_EQ(output_data->rows(), 30);
    ASSERT_EQ(output_data->cols(), 30);
    for (uint32_t c = 0; c < output_data->channels(); ++c) {
      for (uint32_t r = 0; r < output_data->rows(); ++r) {
        for (uint32_t c_ = 0; c_ < output_data->cols(); ++c_) {
          ASSERT_EQ(output_data->at(c, r, c_), 29.);
        }
      }
    }
  }
}

TEST(test_layer, softmax) {
  using namespace kuiper_infer;
  const uint32_t channels = 3;
  const uint32_t rows = 4;
  const uint32_t cols = 4;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, cols, rows);
  input_data->Fill(1.);

  std::vector<std::shared_ptr<Blob>> input_datas;
  input_datas.push_back(input_data);
  input_datas.push_back(input_data);

  SoftmaxLayer softmax_layer;
  std::vector<std::shared_ptr<Blob>> output_datas;
  softmax_layer.Forward(input_datas, output_datas);

  ASSERT_EQ(input_datas.size(), output_datas.size());
  for (uint32_t i = 0; i < input_datas.size(); ++i) {
    std::shared_ptr<Blob> output_data = output_datas.at(i);
    for (uint32_t c = 0; c < output_data->channels(); ++c) {
      for (uint32_t r = 0; r < output_data->rows(); ++r) {
        for (uint32_t c_ = 0; c_ < output_data->cols(); ++c_) {
          ASSERT_EQ(output_data->at(c, r, c_), output_data->at(0, 0, 0));
        }
      }
    }
  }
}

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

//  1 1 1 400

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
  const uint32_t batch = 4;
  const uint32_t channels = 3;
  const uint32_t rows = 4;
  const uint32_t cols = 1;

  std::shared_ptr<Blob> input_data = std::make_shared<Blob>(channels, cols, rows);
  input_data->Fill(1.);
  std::vector<std::shared_ptr<Blob>> input_datas;

  for (int i = 0; i < batch; ++i) {
    input_datas.push_back(input_data);
  }

  std::shared_ptr<Blob> weight_data = std::make_shared<Blob>(channels, rows, cols);
  std::shared_ptr<Blob> bias_data = std::make_shared<Blob>(channels, cols, cols);

  std::vector<std::shared_ptr<Blob>> weight_datas;
  for (int i = 0; i < batch; ++i) {
    weight_datas.push_back(weight_data);
  }

  std::vector<std::shared_ptr<Blob>> bias_datas;
  for (int i = 0; i < batch; ++i) {
    bias_datas.push_back(bias_data);
  }

  std::vector<std::shared_ptr<Blob>> output_datas;

  std::vector<double> weight_datas_(batch * channels * rows * cols, 1.);
  std::vector<double> bias_datas_(batch * channels * cols * cols, 2.);

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

TEST(test_layer, concat) {

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