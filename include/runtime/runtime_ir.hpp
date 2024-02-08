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

/**
 * @brief Runtime representation of a neural network graph
 *
 * This class builds a runtime representation of a neural network
 * computation graph from saved model files. It initializes the graph
 * topology and parameters, sets graph inputs, performs graph execution,
 * and retrieves outputs.
 */
class RuntimeGraph {
 public:
  /**
   * @brief Construct a new RuntimeGraph object
   *
   * Creates a runtime graph representation from parameter and bin files.
   *
   * @param param_path Path to the parameter file defining the graph structure
   * @param bin_path Path to the bin file containing the graph weights
   */
  RuntimeGraph(std::string param_path, std::string bin_path);

  /**
   * @brief Sets the inputs to the graph
   *
   * Sets the input tensors for executing the graph.
   *
   * @param input_name Name of the input
   * @param inputs Vector of input tensors
   */
  void set_inputs(const std::string& input_name, const std::vector<sftensor>& inputs);

  /**
   * @brief Gets output tensors from the graph
   *
   * Returns the output tensors with the given name.
   *
   * @param output_name Name of the graph output
   * @return Vector of output tensors
   */
  std::vector<sftensor> get_outputs(const std::string& output_name) const;

  /**
   * @brief Checks if an op is an input op
   *
   * Returns true if the given op name corresponds to an input operation.
   *
   * @param op_name Name of the op to check
   * @return True if the op is an input op, false otherwise
   */
  bool is_input_op(const std::string& op_name) const;

  /**
   * @brief Checks if an op is an output op
   *
   * Returns true if the given op name is an output operation.
   *
   * @param op_name Name of the op to check
   * @return True if the op is an output op, false otherwise
   */
  bool is_output_op(const std::string& op_name) const;

  /**
   * @brief Builds the runtime graph
   *
   * Performs the actual construction of the runtime graph based on
   * initialized operators and parameters.
   * This must be called after initialization and before execution.
   */
  void Build();

  /**
   * @brief Sets the path to the weights file
   *
   * @param bin_path Path to the weights file
   */
  void set_bin_path(const std::string& bin_path);

  /**
   * @brief Sets the path to the parameter file
   *
   * @param param_path Path to the parameter file
   */
  void set_param_path(const std::string& param_path);

  /**
   * @brief Gets the parameter file path
   *
   * @return The path to the parameter file
   */
  const std::string& param_path() const;

  /**
   * @brief Gets the weights file path
   *
   * @return The path to the weights file
   */
  const std::string& bin_path() const;

  /**
   * @brief Executes the computation graph
   *
   * Executes the graph operations in depth-first order.
   *
   * @param debug Whether to print debugging information during execution
   */
  void Forward(bool debug = false);

 private:
  /**
   * @brief Initializes the graph
   *
   * Initializes the graph from the provided parameter and weights files.
   *
   * @return True if initialization succeeded, false otherwise
   */
  bool Init();

  /**
   * @brief Performs reverse topological sort on the graph
   *
   * Sorts the graph operators in reverse topological order for execution.
   */
  void ReverseTopoSort();

  /**
   * @brief Recursive reverse topological sort
   *
   * Recursive helper for reverse topological sort algorithm.
   *
   * @param root_op Root operator to start sort from
   */
  template <typename T>
  void ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op,
                               int32_t& current_forward_idx);

  /**
   * @brief Creates graph node relations
   *
   * Connects graph operators based on their input and output relations to
   * construct the graph topology.
   */
  void CreateNodeRelation();

  /**
   * @brief Initializes operator inputs
   *
   * Initializes the input tensors of a graph operator.
   *
   * @param inputs Operator inputs in PNNX graph
   * @param runtime_operator The runtime operator to initialize inputs for
   */
  template <typename T>
  static void InitGraphOperatorsInput(
      const std::vector<pnnx::Operand*>& inputs,
      const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

  /**
   * @brief Initializes operator outputs
   *
   * Initializes the output tensors of a graph operator.
   *
   * @param outputs Operator outputs in PNNX graph
   * @param runtime_operator The runtime operator to initialize outputs for
   */
  template <typename T>
  static void InitGraphOperatorsOutput(
      const std::vector<pnnx::Operand*>& outputs,
      const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

  /**
   * @brief Initializes operator attributes
   *
   * Initializes attributes of a graph operator.
   *
   * @param attrs Operator attributes in PNNX graph
   * @param runtime_operator The runtime operator to initialize attributes for
   */
  template <typename T>
  static void InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
                             const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

  /**
   * @brief Initializes operator parameters
   *
   * Initializes the parameters of a graph operator.
   *
   * @param params Operator parameters in PNNX graph
   * @param runtime_operator The runtime operator to initialize parameters for
   */
  template <typename T>
  static void InitGraphParams(const std::map<std::string, pnnx::Parameter>& params,
                              const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator);

  /**
   * @brief Creates a Layer object for a graph operator
   *
   * Creates and initializes a Layer object corresponding to the given
   * RuntimeOperator.
   *
   * @param op The RuntimeOperator to create a Layer for
   * @return Pointer to the created Layer object
   */
  template <typename T>
  static std::shared_ptr<Layer<T>> CreateLayer(const std::shared_ptr<RuntimeOperatorBase<T>>& op);

  /**
   * Propagate output data from current operator to inputs of next operators.
   *
   * @param current_op Current operator whose outputs will be propagated
   * @param layer_output_datas Output data from current operator
   */
  template <typename T>
  static void PropagateLayerOutputs(
      const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
      const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_data);

 private:
  /**
   * @brief Graph state enum
   */
  enum class GraphState {
    NeedInit = -2,
    NeedBuild = -1,
    Complete = 0,
  };

 public:
  /**
   * @brief Gets current graph state
   *
   * @return Current graph state
   */
  GraphState graph_state() const;

 private:
  std::string bin_path_;
  std::string param_path_;
  std::unique_ptr<pnnx::Graph> graph_;

  GraphState graph_state_ = GraphState::NeedInit;
  std::vector<std::shared_ptr<RuntimeOperator>> input_ops_;
  std::vector<std::shared_ptr<RuntimeOperator>> output_ops_;
  std::vector<std::shared_ptr<RuntimeOperator>> operators_;
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_IR_HPP_