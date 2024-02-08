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

// Created by yizhu on 2023/3/20.

#ifndef KUIPER_INFER_TENSOR_UTIL_H
#define KUIPER_INFER_TENSOR_UTIL_H
#include "data/tensor.hpp"

namespace kuiper_infer {

/**
 * @brief Broadcasts two tensors
 *
 * @param tensor1 Tensor 1
 * @param tensor2 Tensor 2
 * @return Tensors with same shape
 */
template <typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> TensorBroadcast(
    const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * @brief Pads a tensor
 *
 * @param tensor Tensor to pad
 * @param pads Padding amount for dims
 * @param padding_value Padding value
 * @return Padded tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorPadding(const std::shared_ptr<Tensor<T>>& tensor,
                                         const std::vector<uint32_t>& pads, T padding_value);

/**
 * @brief Checks tensor equality
 *
 * @param a Tensor 1
 * @param b Tensor 2
 * @param threshold Tolerance
 * @return True if equal within tolerance
 */
template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b,
                  T threshold = 1e-5f);

/**
 * @brief Element-wise tensor add
 *
 * @param tensor1 Tensor 1
 * @param tensor2 Tensor 2
 * @return Output tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                                            const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * @brief In-place element-wise tensor add
 *
 * @param tensor1 Tensor 1
 * @param tensor2 Tensor 2
 * @param output_tensor Output
 */
template <typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                      const std::shared_ptr<Tensor<T>>& tensor2,
                      const std::shared_ptr<Tensor<T>>& output_tensor);

/**
 * @brief Element-wise tensor multiply
 *
 * @param tensor1 Tensor 1
 * @param tensor2 Tensor 2
 * @return Output tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                                                 const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * @brief In-place element-wise tensor multiply
 *
 * @param tensor1 Tensor 1
 * @param tensor2 Tensor 2
 * @param output_tensor Output
 */
template <typename T>
void TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                           const std::shared_ptr<Tensor<T>>& tensor2,
                           const std::shared_ptr<Tensor<T>>& output_tensor);

/**
 * @brief Creates a 3D tensor
 *
 * @param channels Number of channels
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New 3D tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols);

/**
 * @brief Creates a 2D matrix tensor
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @return New 2D tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols);

/**
 * @brief Creates a 1D vector tensor
 *
 * @param size Vector length
 * @return New 1D tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size);

/**
 * @brief Creates tensor with shape
 *
 * @param shapes Tensor dimensions
 * @return New tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(const std::vector<uint32_t>& shapes);

/**
 * @brief Clones a tensor
 *
 * @param tensor Tensor to clone
 * @return Deep copy of tensor
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor);

template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b,
                  T threshold) {
  CHECK(a != nullptr);
  CHECK(b != nullptr);
  if (a->shapes() != b->shapes()) {
    return false;
  }
  bool is_same = arma::approx_equal(a->data(), b->data(), "absdiff", threshold);
  return is_same;
}

template <typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                      const std::shared_ptr<Tensor<T>>& tensor2,
                      const std::shared_ptr<Tensor<T>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
  } else {
    CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
  }
}

template <typename T>
void TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                           const std::shared_ptr<Tensor<T>>& tensor2,
                           const std::shared_ptr<Tensor<T>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    CHECK(tensor1->shapes() == output_tensor->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
  } else {
    CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                                            const std::shared_ptr<Tensor<T>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(tensor1->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    return output_tensor;
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                                                 const std::shared_ptr<Tensor<T>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(tensor1->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] = TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    std::shared_ptr<Tensor<T>> output_tensor = TensorCreate<T>(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    return output_tensor;
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows, uint32_t cols) {
  return std::make_shared<Tensor<T>>(channels, rows, cols);
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols) {
  return std::make_shared<Tensor<T>>(1, rows, cols);
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size) {
  return std::make_shared<Tensor<T>>(1, 1, size);
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);
  if (shapes.size() == 1) {
    return std::make_shared<Tensor<T>>(shapes.at(0));
  } else if (shapes.size() == 2) {
    return std::make_shared<Tensor<T>>(shapes.at(0), shapes.at(1));
  } else {
    return std::make_shared<Tensor<T>>(shapes.at(0), shapes.at(1), shapes.at(2));
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorPadding(const std::shared_ptr<Tensor<T>>& tensor,
                                         const std::vector<uint32_t>& pads, T padding_value) {
  CHECK(tensor != nullptr && !tensor->empty());
  CHECK(pads.size() == 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  std::shared_ptr<Tensor<T>> output =
      std::make_shared<Tensor<T>>(tensor->channels(), tensor->rows() + pad_rows1 + pad_rows2,
                                  tensor->cols() + pad_cols1 + pad_cols2);

  const uint32_t channels = tensor->channels();
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const arma::Mat<T>& in_channel = tensor->slice(channel);
    arma::Mat<T>& output_channel = output->slice(channel);
    const uint32_t in_channel_width = in_channel.n_cols;
    const uint32_t in_channel_height = in_channel.n_rows;

    for (uint32_t w = 0; w < in_channel_width; ++w) {
      T* output_channel_ptr = output_channel.colptr(w + pad_cols1);
      const T* in_channel_ptr = in_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height; ++h) {
        const T value = *(in_channel_ptr + h);
        *(output_channel_ptr + h + pad_rows1) = value;
      }

      for (uint32_t h = 0; h < pad_rows1; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }

      for (uint32_t h = 0; h < pad_rows2; ++h) {
        *(output_channel_ptr + in_channel_height + pad_rows1 + h) = padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols1; ++w) {
      T* output_channel_ptr = output_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols2; ++w) {
      T* output_channel_ptr = output_channel.colptr(pad_cols1 + w + in_channel_width);
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }
  }
  return output;
}

template <typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> TensorBroadcast(
    const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    return {tensor1, tensor2};
  } else {
    CHECK(tensor1->channels() == tensor2->channels());
    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      std::shared_ptr<Tensor<T>> new_tensor =
          TensorCreate<T>(tensor2->channels(), tensor1->rows(), tensor1->cols());
      CHECK(tensor2->size() == tensor2->channels());
      for (uint32_t c = 0; c < tensor2->channels(); ++c) {
        T* new_tensor_ptr = new_tensor->matrix_raw_ptr(c);
        std::fill(new_tensor_ptr, new_tensor_ptr + new_tensor->plane_size(), tensor2->index(c));
      }
      return {tensor1, new_tensor};
    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      std::shared_ptr<Tensor<T>> new_tensor =
          TensorCreate<T>(tensor1->channels(), tensor2->rows(), tensor2->cols());
      CHECK(tensor1->size() == tensor1->channels());
        for (uint32_t c = 0; c < tensor1->channels(); ++c) {
        T* new_tensor_ptr = new_tensor->matrix_raw_ptr(c);
        std::fill(new_tensor_ptr, new_tensor_ptr + new_tensor->plane_size(), tensor1->index(c));
      }
      return {new_tensor, tensor2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {tensor1, tensor2};
    }
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor) {
  return std::make_shared<Tensor<T>>(*tensor);
}
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_TENSOR_UTIL_H
