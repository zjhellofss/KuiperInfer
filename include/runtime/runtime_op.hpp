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
#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "runtime/pnnx/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer {
template <typename T>
class Layer;

/**
 * @brief Base for runtime graph operator
 *
 * Template base class representing an operator node in a runtime graph.
 * Contains node execution order, name, type, layer, inputs, outputs,
 * parameters, attributes etc.
 *
 * @tparam T Operator data type (float, int8, etc.)
 */
template <typename T>
struct RuntimeOperatorBase {
  /// Execution order index of this operator
  int32_t start_time = -1;

  int32_t end_time = -1;

  int32_t occur_end_time = -1;

  /// Whether this operator has run in current execution
  bool has_forward = false;

  /// Name of the operator
  std::string name;

  /// Type of the operator
  std::string type;

  /// Layer for this operator
  std::shared_ptr<Layer<T>> layer;

  /// Names of output operators
  std::vector<std::string> output_names;

  /// Output operand
  std::shared_ptr<RuntimeOperandBase<T>> output_operands;

  /// Input operands mapped by provider name
  std::map<std::string, std::shared_ptr<RuntimeOperandBase<T>>> input_operands;

  /// Input operands in sequence
  std::vector<std::shared_ptr<RuntimeOperandBase<T>>> input_operands_seq;

  /// Output operators mapped by output name
  std::map<std::string, std::shared_ptr<RuntimeOperatorBase<T>>> output_operators;

  /// Operator parameters
  std::map<std::string, std::shared_ptr<RuntimeParameter>> params;

  /// Operator attributes like weights
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
};

using RuntimeOperator = RuntimeOperatorBase<float>;

using RuntimeOperatorQuantized = RuntimeOperatorBase<int8_t>;

template <typename T>
class RuntimeOperatorUtils;

/**
 * @brief Float runtime operator utilities
 *
 * Static utilities for float runtime operators.
 * Initializes operator inputs and outputs.
 */
template <>
class RuntimeOperatorUtils<float> {
 public:
  /**
   * @brief Initializes float operator inputs
   *
   * If first run, initializes input tensors based on shapes.
   * On later runs, checks shape match.
   *
   * @param operators Vector of runtime operators
   */
  static void InitOperatorInput(const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

  /**
   * @brief Initializes float operator outputs
   *
   * If first run, initializes output tensors based on shapes.
   * On later runs, checks shape match.
   *
   * @param pnnx_operators Vector of PNNX operators
   * @param operators Vector of runtime operators
   */
  static void InitOperatorOutput(const std::vector<pnnx::Operator*>& pnnx_operators,
                                 const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
