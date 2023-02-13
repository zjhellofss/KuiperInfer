//
// Created by fss on 22-11-21.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "data/load_data.hpp"

TEST(test_load, load_csv_data) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data1.csv");
  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 3);
  ASSERT_EQ(data.n_cols, 4);
  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      ASSERT_EQ(data.at(i, j), 1);
    }
  }
}

TEST(test_load, load_csv_arange) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data2.csv");
  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 3);
  ASSERT_EQ(data.n_cols, 4);

  int range_data = 0;
  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      ASSERT_EQ(data.at(i, j), range_data);
      range_data += 1;
    }
  }
}

TEST(test_load, load_csv_missing_data1) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data4.csv");
  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 3);
  ASSERT_EQ(data.n_cols, 11);
  int data_one = 0;
  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      if (data.at(i, j) == 1) {
        data_one += 1;
      }
    }
  }
  ASSERT_EQ(data.at(2, 1), 0);
  ASSERT_EQ(data_one, 32);
}

TEST(test_load, load_csv_missing_data2) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data3.csv");

  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 3);
  ASSERT_EQ(data.n_cols, 11);

  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;
  int data_one = 0;
  int data_zero = 0;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      if (data.at(i, j) == 1) {
        data_one += 1;
      } else if (data.at(i, j) == 0) {
        data_zero += 1;
      }
    }
  }
  ASSERT_EQ(data.at(2, 10), 0);
  ASSERT_EQ(data_zero, 1);
  ASSERT_EQ(data_one, 32);
}

TEST(test_load, split_char) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data5.csv", '-');

  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 3);
  ASSERT_EQ(data.n_cols, 11);

  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      ASSERT_EQ(data.at(i, j), 1);
    }
  }
}

TEST(test_load, load_minus_data) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data6.csv", ',');

  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 3);
  ASSERT_EQ(data.n_cols, 11);

  int data_minus_one = 0;

  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      if (data.at(i, j) == -1) {
        data_minus_one += 1;
      }
    }
  }
  ASSERT_EQ(data_minus_one, 33);
}

TEST(test_load, load_large_data) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/data7.csv", ',');
  ASSERT_NE(data.empty(), true);
  ASSERT_EQ(data.n_rows, 1024);
  ASSERT_EQ(data.n_cols, 1024);

  const uint32_t rows = data.n_rows;
  const uint32_t cols = data.n_cols;

  int data_minus_one = 0;
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      if (data.at(i, j) == -1) {
        data_minus_one += 1;
      }
    }
  }
  ASSERT_EQ(data_minus_one, 1024 * 1024);
}

TEST(test_load, load_empty_data) {
  using namespace kuiper_infer;
  const arma::fmat &data = CSVDataLoader::LoadData("./tmp/data_loader/notexists.csv", ',');
  ASSERT_EQ(data.empty(), true);
}