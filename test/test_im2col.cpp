//
// Created by fss on 22-11-22.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/load_data.hpp"
#include "data/tensor.hpp"
#include "data/im2col.hpp"

TEST(test_im2col, test_1c_3w_3h) {
  using namespace kuiper_infer;
  const arma::mat &data1 = CSVDataLoader::LoadData("tmp/im2cols/input.csv");
  const arma::mat &data2 = CSVDataLoader::LoadData("tmp/im2cols/kernel.csv");

  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(3, 5, 5);
  input->at(0) = data1;
  input->at(1) = data1;
  input->at(2) = data1;

  std::shared_ptr<Tensor> kernel = std::make_shared<Tensor>(3, 3, 3);
  kernel->at(0) = data2;
  kernel->at(1) = data2;
  kernel->at(2) = data2;

  const auto &result = Im2Col(input, kernel, 1, 1, 3, 3);
  ASSERT_EQ(result.n_rows, 3);
  ASSERT_EQ(result.n_cols, 3);
  ASSERT_EQ(result.at(0, 0), 1233);
  ASSERT_EQ(result.at(0, 1), 1368);
  ASSERT_EQ(result.at(0, 2), 1503);

  ASSERT_EQ(result.at(1, 0), 1908);
  ASSERT_EQ(result.at(1, 1), 2043);
  ASSERT_EQ(result.at(1, 2), 2178);

  ASSERT_EQ(result.at(2, 0), 2583);
  ASSERT_EQ(result.at(2, 1), 2718);
  ASSERT_EQ(result.at(2, 2), 2853);
}
