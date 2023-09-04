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
#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_IR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_IR_HPP_
#include <glog/logging.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "layer/abstract/layer.hpp"
#include "runtime/pnnx/ir.h"
#include "runtime/runtime_operand.hpp"
#include "runtime_op.hpp"

namespace kuiper_infer {

/// 计算图结构，由多个计算节点和节点之间的数据流图组成
class RuntimeGraph {
 public:
  /**
   * 初始化计算图
   * @param param_path 计算图的结构文件
   * @param bin_path 计算图中的权重文件
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  void set_inputs(const std::string& input_name,
                  const std::vector<sftensor>& inputs);

  std::vector<sftensor> get_outputs(const std::string& output_name) const;

  bool is_input_op(const std::string& op_name) const;

  bool is_output_op(const std::string& op_name) const;

  /**
   * 构建计算图
   * @param input_name 计算图输入节点的名称
   * @param output_name  计算图输出节点的名称
   */
  void Build();

  /**
   * 设置权重文件
   * @param bin_path 权重文件路径
   */
  void set_bin_path(const std::string& bin_path);

  /**
   * 设置结构文件
   * @param param_path  结构文件路径
   */
  void set_param_path(const std::string& param_path);

  /**
   * 返回结构文件
   * @return 返回结构文件
   */
  const std::string& param_path() const;

  /**
   * 返回权重文件
   * @return 返回权重文件
   */
  const std::string& bin_path() const;

  /**
   * 计算图的执行,根据深度优先搜索的顺序执行
   * @param debug 是否调试，如果调试则输出一些中间信息
   */
  void Forward(bool debug = false);

 private:
  /**
   * 计算图的初始化
   * @return 是否初始化成功
   */
  bool Init();

  /**
   * 构建模型的拓扑执行顺序
   * @param root_op 模型的输入节点或根节点，如果有多个输入节点，则调用多次
   */
  void ReverseTopo(const std::shared_ptr<RuntimeOperator>& root_op);

  /**
   * 初始化kuiper infer计算图节点中的输入操作数
   * @param inputs pnnx中的输入操作数
   * @param runtime_operator 计算图节点
   */
  static void InitGraphOperatorsInput(
      const std::vector<pnnx::Operand*>& inputs,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  /**
   * 初始化kuiper infer计算图节点中的输出操作数
   * @param outputs pnnx中的输出操作数
   * @param runtime_operator 计算图节点
   */
  static void InitGraphOperatorsOutput(
      const std::vector<pnnx::Operand*>& outputs,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  /**
   * 初始化kuiper infer计算图中的节点属性
   * @param attrs pnnx中的节点属性
   * @param runtime_operator 计算图节点
   */
  static void InitGraphAttrs(
      const std::map<std::string, pnnx::Attribute>& attrs,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  /**
   * 初始化kuiper infer计算图中的节点参数
   * @param params pnnx中的参数属性
   * @param runtime_operator 计算图节点
   */
  static void InitGraphParams(
      const std::map<std::string, pnnx::Parameter>& params,
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

  /**
   * 根据计算图中的计算节点来返回Layer
   * @param op 计算图中的计算节点
   * @return 创建成功的Layer
   */
  static std::shared_ptr<Layer<float>> CreateLayer(
      const std::shared_ptr<RuntimeOperator>& op);

  /**
   * 探查下一层的计算节点
   * @param current_op 当前计算节点
   * @param layer_output_data 当前节点的输出，赋予到下一层计算节点的输入张量中
   */
  static void ProbeNextLayer(
      const std::shared_ptr<RuntimeOperator>& current_op,
      const std::vector<std::shared_ptr<Tensor<float>>>& layer_output_data);

 private:
  enum class GraphState {
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };

 public:
  /**
   * 返回模型当前的状态
   * @return 返回模型当前的状态
   */
  GraphState graph_state() const;

 private:
  GraphState graph_state_ = GraphState::NeedInit;
  std::vector<std::shared_ptr<RuntimeOperator>>
      input_ops_;  /// 计算图中的输入节点
  std::vector<std::shared_ptr<RuntimeOperator>>
      output_ops_;  /// 计算图中的输出节点

  std::string param_path_;  /// 计算图的结构文件
  std::string bin_path_;    /// 计算图的权重文件

  /// 拓扑排序后的计算图节点
  std::vector<std::shared_ptr<RuntimeOperator>> topo_operators_;
  /// 计算图的计算节点
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
  /// pnnx的graph
  std::unique_ptr<pnnx::Graph> graph_;
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_IR_HPP_