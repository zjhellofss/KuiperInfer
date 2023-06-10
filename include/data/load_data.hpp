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

#ifndef KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#include <armadillo>
#include <string>
namespace kuiper_infer {

class CSVDataLoader {
 public:
  /**
   * 从csv文件中初始化张量
   * @param file_path csv文件的路径
   * @param split_char 分隔符号
   * @return 根据csv文件得到的张量
   */
  static arma::fmat LoadData(const std::string &file_path, char split_char = ',');
 private:
  /**
   * 得到csv文件的尺寸大小，LoadData中根据这里返回的尺寸大小初始化返回的fmat
   * @param file csv文件的路径
   * @param split_char 分割符号
   * @return 根据csv文件的尺寸大小
   */
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
};
}

#endif //KUIPER_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
