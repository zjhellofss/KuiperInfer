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

// Created by fss on 22-11-28.

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
#include <memory>
#include <string>
#include <vector>
#include "data/tensor.hpp"
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace kuiper_infer {
/**
 * @brief Base for runtime graph operand
 *
 * Template base class representing an operand (input/output) in a
 * graph. Contains operand name, shape, data vector, and data type.
 *
 * @tparam T Operand data type (float, int, etc.)
 */
template <typename T>
struct RuntimeOperandBase {
  explicit RuntimeOperandBase() = default;

  explicit RuntimeOperandBase(std::string name, std::vector<int32_t> shapes,
                              std::vector<std::shared_ptr<Tensor<T>>> datas, RuntimeDataType type)
      : name(std::move(name)), shapes(std::move(shapes)), datas(std::move(datas)), type(type) {}

  size_t size() const;

  /// Name of the operand
  std::string name;

  /// Shape of the operand
  std::vector<int32_t> shapes;

  /// Vector containing operand data
  std::vector<std::shared_ptr<Tensor<T>>> datas;

  /// Data type of the operand
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;
};

template <typename T>
size_t RuntimeOperandBase<T>::size() const {
  if (shapes.empty()) {
    return 0;
  }
  size_t size = std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  return size;
}

using RuntimeOperand = RuntimeOperandBase<float>;

using RuntimeOperandQuantized = RuntimeOperandBase<int8_t>;

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERAND_HPP_
