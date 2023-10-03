// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-11-21.
#include "data/load_data.hpp"
#include <glog/logging.h>
#include <armadillo>
#include <fstream>
#include <string>
#include <utility>

namespace kuiper_infer {

std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream& file, char split_char) {
  bool load_ok = file.good();
  file.clear();
  size_t fn_rows = 0;
  size_t fn_cols = 0;
  const std::ifstream::pos_type start_pos = file.tellg();

  std::string token;
  std::string line_str;
  std::stringstream line_stream;

  while (file.good() && load_ok) {
    std::getline(file, line_str);
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
  file.clear();
  file.seekg(start_pos);
  return {fn_rows, fn_cols};
}
}  // namespace kuiper_infer