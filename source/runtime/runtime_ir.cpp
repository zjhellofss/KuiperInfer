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

#include "runtime/runtime_ir.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "layer/abstract/layer_factory.hpp"
#include "runtime/runtime_ir.hpp"
#include "utils/time/time_logging.hpp"

namespace kuiper_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string& bin_path) { this->bin_path_ = bin_path; }

void RuntimeGraph::set_param_path(const std::string& param_path) { this->param_path_ = param_path; }

const std::string& RuntimeGraph::param_path() const { return this->param_path_; }

const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }

static bool IsQuantizeOp(const pnnx::Operator* op) { return false; }

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int32_t load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Can not find the param path or bin path: " << param_path_ << " " << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator*> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  operators_.clear();
  for (const pnnx::Operator* op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node in the model";
      continue;
    } else {
      if (!IsQuantizeOp(op)) {
        std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
        // 初始化算子的名称
        runtime_operator->name = op->name;
        runtime_operator->type = op->type;

        // 初始化算子中的input
        InitGraphOperatorsInput(op->inputs, runtime_operator);

        // 记录输出operand中的名称
        InitGraphOperatorsOutput(op->outputs, runtime_operator);

        // 初始化算子中的attribute(权重)
        InitGraphAttrs(op->attrs, runtime_operator);

        // 初始化算子中的parameter
        InitGraphParams(op->params, runtime_operator);
        this->operators_.push_back(runtime_operator);
      } else {
        LOG(FATAL) << "UnSupported quantize operator in the model " << op->name
                   << " type: " << op->type;
      }
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::Build() {
  if (graph_state_ == GraphState::Complete) {
    LOG(INFO) << "Model has been built already!";
    return;
  }

  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph || graph_state_ == GraphState::NeedInit) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int32_t(graph_state_);
  LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty, may be no init";

  // 构建节点关系
  CreateNodeRelation();

  // 节点拓扑排序
  ReverseTopoSort();

  // 初始化节点的输入和输出空间
  RuntimeOperatorUtils<float>::InitOperatorInput(operators_);
  RuntimeOperatorUtils<float>::InitOperatorOutput(graph_->ops, operators_);

  graph_state_ = GraphState::Complete;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

template <typename T>
StatusCode ExecuteLayer(const std::shared_ptr<Layer<T>>& layer, const std::string& op_name,
                        const std::string& op_type, bool is_debug) {
  CHECK(layer != nullptr);
  StatusCode status;
  if (is_debug) {
    utils::LayerTimeLogging layer_time_logging(op_name, op_type);
    status = layer->Forward();
  } else {
    status = layer->Forward();
  }
  return status;
}

void RuntimeGraph::Forward(bool debug) {
  // 检查当前的执行图是否已经初始化完毕
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!"
               << ", current state is " << int32_t(graph_state_);
  }

  if (debug) {
    utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
  }

  for (const auto& current_op : operators_) {
    current_op->has_forward = false;
    CHECK_GT(current_op->start_time, 0);

    if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
      current_op->has_forward = true;
      continue;
    }

    CHECK(current_op->layer != nullptr)
        << "The layer corresponding to the op " << current_op->name
        << " is empty, indicating that it may not have been created.";

    StatusCode status = ExecuteLayer(current_op->layer, current_op->name, current_op->type, debug);
    CHECK(status == StatusCode::kSuccess)
        << current_op->layer->layer_name()
        << " layer forward failed, error code: " << int32_t(status);

    current_op->has_forward = true;
    PropagateLayerOutputs(current_op, current_op->output_operands->datas);
  }

  if (debug) {
    utils::LayerTimeLogging::SummaryLogging();
  }

  for (const auto& op : operators_) {
    LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
  }
}

template <typename T>
std::shared_ptr<Layer<T>> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperatorBase<T>>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  auto layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}

template <typename T>
void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (inputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* input : inputs) {
    if (!input) {
      continue;
    }

    std::vector<int32_t> dims;
    const pnnx::Operator* producer = input->producer;

    for (int32_t dim : input->shape) {
      dims.push_back(dim);
    }
    CHECK(!dims.empty());
    std::shared_ptr<RuntimeOperandBase<T>> runtime_operand =
        std::make_shared<RuntimeOperandBase<T>>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = dims;
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 7: {
        runtime_operand->type = RuntimeDataType::kTypeInt8;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (outputs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const pnnx::Operand* output : outputs) {
    if (!output) {
      continue;
    }
    const auto& consumers = output->consumers;
    for (const auto& c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (params.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const auto& [name, parameter] : params) {
    const int32_t type = parameter.type;
    switch (type) {
      case int32_t(RuntimeParameterType::kParameterUnknown): {
        std::shared_ptr<RuntimeParameter> runtime_parameter = std::make_shared<RuntimeParameter>();
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterBool): {
        std::shared_ptr<RuntimeParameterBool> runtime_parameter =
            std::make_shared<RuntimeParameterBool>(parameter.b);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterInt): {
        std::shared_ptr<RuntimeParameterInt> runtime_parameter =
            std::make_shared<RuntimeParameterInt>(parameter.i);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterFloat): {
        std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
            std::make_shared<RuntimeParameterFloat>(parameter.f);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterString): {
        std::shared_ptr<RuntimeParameterString> runtime_parameter =
            std::make_shared<RuntimeParameterString>(parameter.s);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterIntArray): {
        std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
            std::make_shared<RuntimeParameterIntArray>(parameter.ai);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int32_t(RuntimeParameterType::kParameterFloatArray): {
        std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
            std::make_shared<RuntimeParameterFloatArray>(parameter.af);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int32_t(RuntimeParameterType::kParameterStringArray): {
        std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
            std::make_shared<RuntimeParameterStringArray>(parameter.as);
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type: " << type;
      }
    }
  }
}

template <typename T>
void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
                                  const std::shared_ptr<RuntimeOperatorBase<T>>& runtime_operator) {
  if (attrs.empty()) {
    return;
  }
  CHECK(runtime_operator != nullptr) << "The runtime operator is null pointer";
  for (const auto& [name, attr] : attrs) {
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>(
            attr.shape, RuntimeDataType::kTypeFloat32, attr.data);
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown attribute type: " << attr.type;
      }
    }
  }
}

template <typename T>
void RuntimeGraph::PropagateLayerOutputs(
    const std::shared_ptr<RuntimeOperatorBase<T>>& current_op,
    const std::vector<std::shared_ptr<Tensor<T>>>& layer_output_datas) {
  // For each next operator of current operator
  for (const auto& [_, output_op] : current_op->output_operators) {
    // Get next op's input operands corresponding to current op's output
    const auto& next_input_operands = output_op->input_operands;
    const auto& next_input_op_iter = next_input_operands.find(current_op->name);
    if (next_input_op_iter != next_input_operands.end()) {
      // Get input data spaces for those operands
      std::vector<stensor<T>>& next_input_datas = next_input_op_iter->second->datas;
      // Copy current op output data to next op input data
      for (uint32_t i = 0; i < next_input_datas.size(); ++i) {
        const stensor<T>& layer_output_data = layer_output_datas.at(i);
        if (next_input_datas.at(i) != nullptr) {
          CHECK(next_input_datas.at(i)->shapes() == layer_output_data->shapes());
        }
        next_input_datas.at(i) = layer_output_data;
      }
    }
  }
}

void RuntimeGraph::ReverseTopoSort() {
  // 构建拓扑顺序
  for (const auto& op : operators_) {
    // 根据输入节点构建拓扑排序
    if (op != nullptr && !op->has_forward) {
      int32_t current_forward_idx = 0;
      this->ReverseTopoSortInternal(op, current_forward_idx);
    }
  }

  // 根据拓扑顺序调整算子的执行顺序
  std::sort(operators_.begin(), operators_.end(), [](const auto& op1, const auto& op2) {
    return op1->start_time > op2->start_time;
  });

  int32_t forward_index = 1;
  for (const auto& op : operators_) {
    op->start_time = forward_index;
    forward_index += 1;
  }

  for (const auto& op : operators_) {
    const auto& next_ops = op->output_operators;
    int32_t last_forward_index = -1;
    for (const auto& [_, next_op] : next_ops) {
      if (next_op->start_time >= last_forward_index) {
        last_forward_index = next_op->start_time;
      }
    }

    if (last_forward_index == -1) {
      op->end_time = op->start_time + 1;
    } else {
      op->end_time = last_forward_index;
    }
    op->occur_end_time = -1;
  }
}

template <typename T>
void RuntimeGraph::ReverseTopoSortInternal(const std::shared_ptr<RuntimeOperatorBase<T>>& root_op,
                                           int32_t& current_forward_idx) {
  if (!root_op) {
    LOG(INFO) << "Current operator is nullptr";
    return;
  }
  if (root_op->input_operands.empty() && !root_op->has_forward) {
    this->input_ops_.push_back(root_op);
  }
  if (root_op->output_names.empty() && !root_op->has_forward) {
    this->output_ops_.push_back(root_op);
  }

  root_op->has_forward = true;
  const auto& next_ops = root_op->output_operators;
  for (const auto& [_, op] : next_ops) {
    if (op != nullptr && !op->has_forward) {
      this->ReverseTopoSortInternal(op, current_forward_idx);
    }
  }

  for (const auto& [_, op] : next_ops) {
    CHECK_EQ(op->has_forward, true);
  }
  root_op->start_time = current_forward_idx;
  current_forward_idx += 1;
}

void RuntimeGraph::CreateNodeRelation() {
  // 构建图关系
  for (const auto& current_op : this->operators_) {
    // 获取当前节点的所有后继节点的names，遍历根据next_op_name从operators_maps_中插入所需要的节点
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& kOutputName : output_names) {
      for (const auto& output_op : this->operators_) {
        if (output_op != current_op && output_op->name == kOutputName) {
          current_op->output_operators.insert({kOutputName, output_op});
        }
      }
    }
    // 除了输入和输出节点，都创建layer
    if (current_op->type != "pnnx.Input" && current_op->type != "pnnx.Output") {
      auto layer = RuntimeGraph::CreateLayer(current_op);
      if (layer) {
        current_op->layer = layer;
        layer->set_runtime_operator(current_op);
      } else {
        LOG(FATAL) << "Layer " << current_op->name << " create failed!";
      }
    }
  }
}

RuntimeGraph::GraphState RuntimeGraph::graph_state() const { return this->graph_state_; }

void RuntimeGraph::set_inputs(const std::string& input_name, const std::vector<sftensor>& inputs) {
  CHECK(this->graph_state_ == GraphState::Complete);
  std::shared_ptr<RuntimeOperator> input_op;
  for (auto op : this->input_ops_) {
    if (op->name == input_name) {
      input_op = op;
      break;
    }
  }
  CHECK(input_op != nullptr) << "Can not find the input operator: " << input_name;
  PropagateLayerOutputs(input_op, inputs);
}

std::vector<sftensor> RuntimeGraph::get_outputs(const std::string& output_name) const {
  CHECK(this->graph_state_ == GraphState::Complete);
  std::shared_ptr<RuntimeOperator> output_op;
  for (auto op : this->output_ops_) {
    if (op->name == output_name) {
      output_op = op;
    }
  }

  CHECK(output_op != nullptr) << "Can not find the output operator: " << output_name;
  std::vector<sftensor> outputs;
  for (const auto& input_operand : output_op->input_operands_seq) {
    std::copy(input_operand->datas.begin(), input_operand->datas.end(),
              std::back_inserter(outputs));
  }
  return outputs;
}

bool RuntimeGraph::is_input_op(const std::string& op_name) const {
  for (auto op : this->input_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

bool RuntimeGraph::is_output_op(const std::string& op_name) const {
  for (auto op : this->output_ops_) {
    CHECK(op != nullptr);
    if (op->name == op_name) {
      return true;
    }
  }
  return false;
}

}  // namespace kuiper_infer
