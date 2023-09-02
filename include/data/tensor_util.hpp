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
 * 对张量进行形状上的扩展
 * @param tenor1 张量1
 * @param tensor2 张量2
 * @return 形状一致的张量
 */
template <typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>>
TensorBroadcast(const std::shared_ptr<Tensor<T>>& tensor1,
                const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * 对张量的填充
 * @param tensor 待填充的张量
 * @param pads 填充的大小
 * @param padding_value 填充的值
 * @return 填充之后的张量
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorPadding(
    const std::shared_ptr<Tensor<T>>& tensor, const std::vector<uint32_t>& pads,
    T padding_value);

/**
 * 比较tensor的值是否相同
 * @param a 输入张量1
 * @param b 输入张量2
 * @param threshold 张量之间差距的阈值
 * @return 比较结果
 */
template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a,
                  const std::shared_ptr<Tensor<T>>& b, T threshold = 1e-5f);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相加的结果
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(
    const std::shared_ptr<Tensor<T>>& tensor1,
    const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
template <typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                      const std::shared_ptr<Tensor<T>>& tensor2,
                      const std::shared_ptr<Tensor<T>>& output_tensor);

/**
 * 矩阵点乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
template <typename T>
void TensorElementMultiply(const std::shared_ptr<Tensor<T>>& tensor1,
                           const std::shared_ptr<Tensor<T>>& tensor2,
                           const std::shared_ptr<Tensor<T>>& output_tensor);

/**
 * 张量相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相乘的结果
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(
    const std::shared_ptr<Tensor<T>>& tensor1,
    const std::shared_ptr<Tensor<T>>& tensor2);

/**
 * 创建一个张量
 * @param channels 通道数量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows,
                                        uint32_t cols);

/**
 * 创建一个张量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t rows, uint32_t cols);

/**
 * 创建一个张量
 * @param size 数据数
 * @return 创建后的张量
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t size);

/**
 * 创建一个张量
 * @param shapes 张量的形状
 * @return 创建后的张量
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(const std::vector<uint32_t>& shapes);

/**
 * 返回一个深拷贝后的张量
 * @param 待Clone的张量
 * @return 新的张量
 */
template <typename T>
std::shared_ptr<Tensor<T>> TensorClone(std::shared_ptr<Tensor<T>> tensor);

template <typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a,
                  const std::shared_ptr<Tensor<T>>& b, T threshold) {
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
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
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
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(
    const std::shared_ptr<Tensor<T>>& tensor1,
    const std::shared_ptr<Tensor<T>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    std::shared_ptr<Tensor<T>> output_tensor =
        TensorCreate<T>(tensor1->shapes());
    output_tensor->set_data(tensor1->data() + tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    std::shared_ptr<Tensor<T>> output_tensor =
        TensorCreate<T>(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    return output_tensor;
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorElementMultiply(
    const std::shared_ptr<Tensor<T>>& tensor1,
    const std::shared_ptr<Tensor<T>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    std::shared_ptr<Tensor<T>> output_tensor =
        TensorCreate<T>(tensor1->shapes());
    output_tensor->set_data(tensor1->data() % tensor2->data());
    return output_tensor;
  } else {
    // broadcast
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    std::shared_ptr<Tensor<T>> output_tensor =
        TensorCreate<T>(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    return output_tensor;
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorCreate(uint32_t channels, uint32_t rows,
                                        uint32_t cols) {
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
    return std::make_shared<Tensor<T>>(shapes.at(0), shapes.at(1),
                                       shapes.at(2));
  }
}

template <typename T>
std::shared_ptr<Tensor<T>> TensorPadding(
    const std::shared_ptr<Tensor<T>>& tensor, const std::vector<uint32_t>& pads,
    T padding_value) {
  CHECK(tensor != nullptr && !tensor->empty());
  CHECK(pads.size() == 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  std::shared_ptr<Tensor<T>> output = std::make_shared<Tensor<T>>(
      tensor->channels(), tensor->rows() + pad_rows1 + pad_rows2,
      tensor->cols() + pad_cols1 + pad_cols2);

  const uint32_t channels = tensor->channels();
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const arma::Mat<T>& in_channel = tensor->slice(channel);
    arma::Mat<T>& output_channel = output->slice(channel);
    const uint32_t in_channel_width = in_channel.n_cols;
    const uint32_t in_channel_height = in_channel.n_rows;

    for (uint32_t w = 0; w < in_channel_width; ++w) {
      T* output_channel_ptr =
          const_cast<T*>(output_channel.colptr(w + pad_cols1));
      const T* in_channel_ptr = in_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height; ++h) {
        const T value = *(in_channel_ptr + h);
        *(output_channel_ptr + h + pad_rows1) = value;
      }

      for (uint32_t h = 0; h < pad_rows1; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }

      for (uint32_t h = 0; h < pad_rows2; ++h) {
        *(output_channel_ptr + in_channel_height + pad_rows1 + h) =
            padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols1; ++w) {
      T* output_channel_ptr = const_cast<T*>(output_channel.colptr(w));
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols2; ++w) {
      T* output_channel_ptr = const_cast<T*>(
          output_channel.colptr(pad_cols1 + w + in_channel_width));
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }
  }
  return output;
}

template <typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>>
TensorBroadcast(const std::shared_ptr<Tensor<T>>& tensor1,
                const std::shared_ptr<Tensor<T>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    return {tensor1, tensor2};
  } else {
    CHECK(tensor1->channels() == tensor2->channels());
    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      std::shared_ptr<Tensor<T>> new_tensor = TensorCreate<T>(
          tensor2->channels(), tensor1->rows(), tensor1->cols());
      CHECK(tensor2->size() == tensor2->channels());
      for (uint32_t c = 0; c < tensor2->channels(); ++c) {
        new_tensor->slice(c).fill(tensor2->index(c));
      }
      return {tensor1, new_tensor};
    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      std::shared_ptr<Tensor<T>> new_tensor = TensorCreate<T>(
          tensor1->channels(), tensor2->rows(), tensor2->cols());
      CHECK(tensor1->size() == tensor1->channels());
      for (uint32_t c = 0; c < tensor1->channels(); ++c) {
        new_tensor->slice(c).fill(tensor1->index(c));
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
