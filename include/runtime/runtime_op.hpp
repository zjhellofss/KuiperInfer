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
// #include "layer/abstract/layer.hpp"
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer {
class Layer;

/// 计算图中的计算节点
struct RuntimeOperator {
  bool has_forward = false;
  std::string name;              /// 计算节点的名称
  std::string type;              /// 计算节点的类型
  std::shared_ptr<Layer> layer;  /// 节点对应的计算Layer

  std::vector<std::string> output_names;  /// 节点的输出节点名称
  std::shared_ptr<RuntimeOperand> output_operands;  /// 节点的输出操作数

  std::map<std::string, std::shared_ptr<RuntimeOperand>>
      input_operands;  /// 节点的输入操作数
  std::vector<std::shared_ptr<RuntimeOperand>>
      input_operands_seq;  /// 节点的输入操作数，顺序排列
  std::map<std::string, std::shared_ptr<RuntimeOperator>>
      output_operators;  /// 输出节点的名字和节点对应

  std::map<std::string, std::shared_ptr<RuntimeParameter>>
      params;  /// 算子的参数信息
  std::map<std::string, std::shared_ptr<RuntimeAttribute>>
      attribute;  /// 算子的属性信息，内含权重信息
};

class RuntimeOperatorUtils {
 public:
  /**
   * 如果图是第一次运行，则根据节点输入operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输入operand的形状和operand中张量的形状是否匹配
   * @param operators 计算图中的计算节点
   */
  static void InitOperatorInput(
      const std::vector<std::shared_ptr<RuntimeOperator>>& operators);

  /**
   * 如果图是第一次运行，则根据节点输出operand的形状准备好后续Layer计算中所需要的Tensor
   * 如果图是第二次以上运行，则检查输出operand的形状和operand中张量的形状是否匹配
   * @param pnnx_operators pnnx图节点
   * @param operators KuiperInfer计算图中的计算节点
   */
  static void InitOperatorOutput(
      const std::vector<pnnx::Operator*>& pnnx_operators,
      const std::vector<std::shared_ptr<RuntimeOperator>>& operators);
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
