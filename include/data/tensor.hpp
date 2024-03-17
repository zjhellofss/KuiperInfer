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

// Created by fss on 22-11-12.

#ifndef KUIPER_INFER_DATA_BLOB_HPP_
#define KUIPER_INFER_DATA_BLOB_HPP_
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>

namespace kuiper_infer {
template <typename T>
class Tensor {
 public:
  explicit Tensor(T* raw_ptr, uint32_t size);

  explicit Tensor(T* raw_ptr, uint32_t rows, uint32_t cols);

  explicit Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols);

  explicit Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes);

  /**
   * @brief Construct a new empty Tensor
   */
  explicit Tensor() = default;

  /**
   * @brief Construct a 3D Tensor
   *
   * @param channels Number of channels
   * @param rows Number of rows
   * @param cols Number of columns
   */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  /**
   * @brief Construct a 1D Tensor
   *
   * @param size Vector size
   */
  explicit Tensor(uint32_t size);

  /**
   * @brief Construct a 2D matrix Tensor
   *
   * @param rows Number of rows
   * @param cols Number of columns
   */
  explicit Tensor(uint32_t rows, uint32_t cols);

  /**
   * @brief Construct Tensor with shape
   *
   * @param shapes Tensor dimensions
   */
  explicit Tensor(const std::vector<uint32_t>& shapes);

  /**
   * @brief Gets number of rows
   *
   * @return Number of rows
   */
  uint32_t rows() const;

  /**
   * @brief Gets number of columns
   *
   * @return Number of columns
   */
  uint32_t cols() const;

  /**
   * @brief Gets number of channels
   *
   * @return Number of channels
   */
  uint32_t channels() const;

  /**
   * @brief Gets total number of elements
   *
   * @return Total number of elements
   */
  size_t size() const;

  size_t plane_size() const;

  /**
   * @brief Sets the tensor data
   *
   * @param data Data to set
   */
  void set_data(const arma::Cube<T>& data);

  /**
   * @brief Checks if tensor is empty
   *
   * @return True if empty, false otherwise
   */
  bool empty() const;

  /**
   * @brief Gets element reference at offset
   *
   * @param offset Element offset
   * @return Element reference
   */
  T& index(uint32_t offset);

  /**
   * @brief Gets element at offset
   *
   * @param offset Element offset
   * @return Element value
   */
  const T index(uint32_t offset) const;

  /**
   * @brief Gets tensor shape
   *
   * @return Tensor dimensions
   */
  std::vector<uint32_t> shapes() const;

  /**
   * @brief Gets raw tensor shape
   *
   * @return Raw tensor dimensions
   */
  const std::vector<uint32_t>& raw_shapes() const;

  /**
   * @brief Gets tensor data
   *
   * @return Tensor data
   */
  arma::Cube<T>& data();

  /**
   * @brief Gets tensor data (const)
   *
   * @return Tensor data
   */
  const arma::Cube<T>& data() const;

  /**
   * @brief Gets channel matrix
   *
   * @param channel Channel index
   * @return Channel matrix
   */
  arma::Mat<T>& slice(uint32_t channel);

  /**
   * @brief Gets channel matrix (const)
   *
   * @param channel Channel index
   * @return Channel matrix
   */
  const arma::Mat<T>& slice(uint32_t channel) const;

  /**
   * @brief Gets element at location
   *
   * @param channel Channel
   * @param row Row index
   * @param col Column index
   * @return Element at location
   */
  const T at(uint32_t channel, uint32_t row, uint32_t col) const;

  /**
   * @brief Gets element reference at location
   *
   * @param channel Channel
   * @param row Row index
   * @param col Column index
   * @return Element reference
   */
  T& at(uint32_t channel, uint32_t row, uint32_t col);

  /**
   * @brief Pads the tensor
   *
   * @param pads Padding amount for dimensions
   * @param padding_value Padding value
   */
  void Padding(const std::vector<uint32_t>& pads, T padding_value);

  /**
   * @brief Fills tensor with value
   *
   * @param value Fill value
   */
  void Fill(T value);

  /**
   * @brief Fills tensor with vector data
   *
   * @param values Data to fill
   * @param row_major Fill by row-major order
   */
  void Fill(const std::vector<T>& values, bool row_major = true);

  /**
   * @brief Gets tensor data as vector
   *
   * @param row_major Row-major or column-major order
   * @return Tensor data
   */
  std::vector<T> values(bool row_major = true);

  /**
   * @brief Initializes with ones
   */
  void Ones();

  /**
   * @brief Initializes with normal distribution
   *
   * @param mean Mean value
   * @param var Variance
   */
  void RandN(T mean = 0, T var = 1);

  /**
   * @brief Initializes with uniform distribution
   *
   * @param min Minimum value
   * @param max Maximum value
   */
  void RandU(T min = 0, T max = 1);

  /**
   * @brief Prints tensor
   */
  void Show();

  /**
   * @brief Reshape tensor
   *
   * @param shapes New shape
   * @param row_major Row-major or column-major
   */
  void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

  /**
   * @brief Flattens the tensor
   *
   * @param row_major Row-major or column-major
   */
  void Flatten(bool row_major = false);

  /**
   * @brief Applies element-wise transform
   *
   * @param filter Transform function
   */
  void Transform(const std::function<T(T)>& filter);

  /**
   * @brief Gets raw data pointer
   *
   * @return Raw data pointer
   */
  T* raw_ptr();

  /**
   * @brief Gets raw data pointer
   *
   * @return Raw data pointer
   */
  const T* raw_ptr() const;

  /**
   * @brief Gets raw data pointer with offset
   *
   * @param offset Offset
   * @return Raw pointer + offset
   */
  T* raw_ptr(size_t offset);

  /**
   * @brief Gets raw data pointer with offset
   *
   * @param offset Offset
   * @return Raw pointer + offset
   */
  const T* raw_ptr(size_t offset) const;

  /**
   * @brief Gets matrix raw pointer
   *
   * @param index Matrix index
   * @return Raw pointer to matrix
   */
  T* matrix_raw_ptr(uint32_t index);

  /**
   * @brief Gets matrix raw pointer
   *
   * @param index Matrix index
   * @return Raw pointer to matrix
   */
  const T* matrix_raw_ptr(uint32_t index) const;

 private:
  /**
   * @brief Checks tensor shape
   *
   * @param shapes Tensor shape
   */
  void Review(const std::vector<uint32_t>& shapes);

  /// Raw tensor dimensions
  std::vector<uint32_t> raw_shapes_;

  /// Tensor data
  arma::Cube<T> data_;
};

template <typename T = float>
using stensor = std::shared_ptr<Tensor<T>>;

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

using u1tensor = Tensor<uint8_t>;
using su1tensor = std::shared_ptr<Tensor<uint8_t>>;

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_DATA_BLOB_HPP_
