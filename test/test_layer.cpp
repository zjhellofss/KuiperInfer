//
// Created by fss on 22-11-12.
//
#include <gtest/gtest.h>
#include "../source/layer/avgpooling.hpp"

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