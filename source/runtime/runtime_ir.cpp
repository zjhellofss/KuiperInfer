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
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
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
    LOG_IF(FATAL, !init_graph) << "Init graph failed!";
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

void RuntimeGraph::Forward(bool debug) {
  using namespace utils;
  // 检查当前的执行图是否已经初始化完毕
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!";
  }
  CHECK(graph_state_ == GraphState::Complete)
      << "Graph status error, current state is " << int32_t(graph_state_);

  for (const auto& op : operators_) {
    op->has_forward = false;
    CHECK_GT(op->forward_index, 0);
  }

  if (debug) {
    LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
  }

  auto forward_layer = [&](const std::shared_ptr<Layer<float>>& layer, const std::string& op_name,
                           const std::string& op_type) {
    StatusCode status;
    if (debug) {
      {
        LayerTimeLogging layer_time_logging(op_name, op_type);
        status = layer->Forward();
      }
    } else {
      status = layer->Forward();
    }
    return status;
  };

  for (const auto& current_op : operators_) {
    if (is_input_op(current_op->name) || is_output_op(current_op->name)) {
      current_op->has_forward = true;
      continue;
    }

    CHECK(current_op->layer != nullptr)
        << "The layer corresponding to the op " << current_op->name
        << " is empty, indicating that it may not have been created.";

    std::shared_ptr<Layer<float>> layer = current_op->layer;
    StatusCode status = forward_layer(layer, current_op->name, current_op->type);
    CHECK(status == StatusCode::kSuccess)
        << layer->layer_name() << " layer forward failed, error code: " << int32_t(status);

    current_op->has_forward = true;
    ProbeNextLayer(current_op, current_op->output_operands->datas);
  }

  if (debug) {
    LayerTimeLogging::SummaryLogging();
  }

  for (const auto& op : operators_) {
    LOG_IF(FATAL, !op->has_forward) << "The operator: " << op->name << " has not been forward yet!";
  }
}

std::shared_ptr<Layer<float>> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  auto layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}

void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  if (inputs.empty()) {
    return;
  }

  for (const pnnx::Operand* input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator* producer = input->producer;
    std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();

    runtime_operand->name = producer->name;
    for (int32_t dim : input->shape) {
      runtime_operand->shapes.push_back(dim);
    }
    CHECK(!runtime_operand->shapes.empty());

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      case 0: {
        runtime_operand->type = RuntimeDataType::kTypeUnknown;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }

    runtime_operator->input_operands.insert({producer->name, runtime_operand});
    runtime_operator->input_operands_seq.push_back(runtime_operand);
  }
}

void RuntimeGraph::InitGraphOperatorsOutput(
    const std::vector<pnnx::Operand*>& outputs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  if (outputs.empty()) {
    return;
  }

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

void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter>& params,
                                   const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  if (params.empty()) {
    return;
  }

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

void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute>& attrs,
                                  const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  if (attrs.empty()) {
    return;
  }
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

void RuntimeGraph::ProbeNextLayer(const std::shared_ptr<RuntimeOperator>& current_op,
                                  const std::vector<sftensor>& layer_output_datas) {
  // For each next operator of current operator
  const auto& next_ops = current_op->output_operators;
  for (const auto& [_, next_op] : next_ops) {
    // Get next op's input operands corresponding to current op's output
    const auto& next_input_operands = next_op->input_operands;
    const auto& next_input_op_iter = next_input_operands.find(current_op->name);
    if (next_input_op_iter != next_input_operands.end()) {
      // Get input data spaces for those operands
      std::vector<sftensor>& next_input_datas = next_input_op_iter->second->datas;
      // Copy current op output data to next op input data
      for (uint32_t i = 0; i < next_input_datas.size(); ++i) {
        sftensor layer_output_data = layer_output_datas.at(i);
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
  start_forward_index_ = 0;
  for (const auto& op : operators_) {
    // 根据输入节点构建拓扑排序
    if (!op->has_forward) {
      this->ReverseTopoSort_(op);
    }
  }

  // 根据拓扑顺序调整算子的执行顺序
  std::sort(operators_.begin(), operators_.end(), [](const auto& op1, const auto& op2) {
    return op1->forward_index > op2->forward_index;
  });

  int32_t forward_index = 1;
  for (const auto& op : operators_) {
    op->forward_index = forward_index;
    forward_index += 1;
  }
}

void RuntimeGraph::ReverseTopoSort_(const std::shared_ptr<RuntimeOperator>& root_op) {
  CHECK(root_op != nullptr) << "current operator is nullptr";
  if (root_op->input_operands.empty() && !root_op->has_forward) {
    this->input_ops_.push_back(root_op);
  }
  if (root_op->output_names.empty() && !root_op->has_forward) {
    this->output_ops_.push_back(root_op);
  }

  root_op->has_forward = true;
  const auto& next_ops = root_op->output_operators;
  for (const auto& [_, op] : next_ops) {
    if (op != nullptr) {
      if (!op->has_forward) {
        this->ReverseTopoSort_(op);
      }
    }
  }

  for (const auto& [_, op] : next_ops) {
    CHECK_EQ(op->has_forward, true);
  }
  root_op->forward_index = start_forward_index_;
  start_forward_index_ += 1;
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
      std::shared_ptr<Layer<float>> layer = RuntimeGraph::CreateLayer(current_op);
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
  ProbeNextLayer(input_op, inputs);
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
  CHECK(output_op->input_operands_seq.size() == 1)
      << "KuiperInfer only supports one output per operator.";

  output_op->output_operands = output_op->input_operands_seq.front();
  CHECK(output_op->output_operands != nullptr) << "Output from " << output_op->name << " is empty";
  return output_op->output_operands->datas;
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
