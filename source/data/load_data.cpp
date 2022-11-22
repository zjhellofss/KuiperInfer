//
// Created by fss on 22-11-21.
//
#include "data/load_data.hpp"
#include <string>
#include <fstream>
#include <armadillo>
#include <utility>
#include <glog/logging.h>

namespace kuiper_infer {

arma::mat CSVDataLoader::LoadData(const std::string &file_path, const char split_char) {
  CHECK(!file_path.empty()) << "File path is empty!";
  std::ifstream in(file_path);
  CHECK(in.is_open() && in.good()) << "File open failed! " << file_path;

  arma::mat data;
  std::string line_str;
  std::stringstream line_stream;

  const auto &[rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
  data.zeros(rows, cols);

  size_t row = 0;
  while (in.good()) {
    std::getline(in, line_str);
    if (line_str.empty()) {
      LOG(ERROR) << "Meet a empty line!";
      break;
    }

    std::string token;
    line_stream.clear();
    line_stream.str(line_str);

    size_t col = 0;
    while (line_stream.good()) {
      std::getline(line_stream, token, split_char);
      try {
        data.at(row, col) = std::stod(token);
      }
      catch (...) {
        continue;
      }
      col += 1;
      CHECK(col <= cols) << "There are excessive elements on the column";
    }

    row += 1;
    CHECK(row <= rows) << "There are excessive elements on the row";
  }
  return data;
}

std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream &f, char split_char) {
  bool load_ok = f.good();
  f.clear();
  size_t fn_rows = 0;
  size_t fn_cols = 0;
  const std::ifstream::pos_type start_pos = f.tellg();

  std::string token;
  std::string line_str;
  std::stringstream line_stream;

  while (f.good() && load_ok) {
    std::getline(f, line_str);
    if (line_str.empty()) {
      break;
    }

    line_stream.clear();
    line_stream.str(line_str);
    size_t line_cols = 0;

    std::string row_token;
    while (line_stream.good()) {
      std::getline(line_stream, row_token, split_char);
      ++line_cols;
    }
    if (line_cols > fn_cols) {
      fn_cols = line_cols;
    }

    ++fn_rows;
  }
  f.clear();
  f.seekg(start_pos);
  return {fn_rows, fn_cols};
}
}