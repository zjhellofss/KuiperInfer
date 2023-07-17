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
#include "utils/time_logging.hpp"

namespace kuiper_infer {
RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
    : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

void RuntimeGraph::set_bin_path(const std::string& bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string& param_path) {
  this->param_path_ = param_path;
}

const std::string& RuntimeGraph::param_path() const {
  return this->param_path_;
}

const std::string& RuntimeGraph::bin_path() const { return this->bin_path_; }

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
               << " " << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator*> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
  this->operators_maps_.clear();
  for (const pnnx::Operator* op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
      std::shared_ptr<RuntimeOperator> runtime_operator =
          std::make_shared<RuntimeOperator>();
      // 初始化算子的名称
      runtime_operator->name = op->name;
      runtime_operator->type = op->type;

      // 初始化算子中的input
      const std::vector<pnnx::Operand*>& inputs = op->inputs;
      if (!inputs.empty()) {
        InitGraphOperatorsInput(inputs, runtime_operator);
      }

      // 记录输出operand中的名称
      const std::vector<pnnx::Operand*>& outputs = op->outputs;
      if (!outputs.empty()) {
        InitGraphOperatorsOutput(outputs, runtime_operator);
      }

      // 初始化算子中的attribute(权重)
      const std::map<std::string, pnnx::Attribute>& attrs = op->attrs;
      if (!attrs.empty()) {
        InitGraphAttrs(attrs, runtime_operator);
      }

      // 初始化算子中的parameter
      const std::map<std::string, pnnx::Parameter>& params = op->params;
      if (!params.empty()) {
        InitGraphParams(params, runtime_operator);
      }
      this->operators_.push_back(runtime_operator);
      this->operators_maps_.insert({runtime_operator->name, runtime_operator});
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::Build(const std::string& input_name,
                         const std::string& output_name) {
  if (graph_state_ == GraphState::Complete) {
    LOG(INFO) << "Model has been built already!";
    return;
  }

  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int(graph_state_);
  LOG_IF(FATAL, this->operators_.empty())
      << "Graph operators is empty, may be no init";

  // 构建图关系
  for (const auto& current_op : this->operators_) {
    // 获取当前节点的所有后继节点的names，遍历根据next_op_name从operators_maps_中插入所需要的节点
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& kOutputName : output_names) {
      if (const auto& output_op = this->operators_maps_.find(kOutputName);
          output_op != this->operators_maps_.end()) {
        current_op->output_operators.insert({kOutputName, output_op->second});
      }
    }
  }

  for (const auto& kOperator : this->operators_) {
    // 除了输入和输出节点，都创建layer
    if (kOperator->type != "pnnx.Input" && kOperator->type != "pnnx.Output") {
      std::shared_ptr<Layer> layer = RuntimeGraph::CreateLayer(kOperator);
      CHECK(layer != nullptr)
          << "Layer " << kOperator->name << " create failed!";
      if (layer) {
        kOperator->layer = layer;
        layer->set_runtime_operator(kOperator);
      }
    }
  }

  // 初始化节点的输入和输出空间
  RuntimeOperatorUtils::InitOperatorInput(operators_);
  RuntimeOperatorUtils::InitOperatorOutput(graph_->ops, operators_);

  // 构建拓扑顺序
  topo_operators_.clear();
  for (const auto& [_, op] : operators_maps_) {
    // 根据输入节点构建拓扑排序
    if (op->type == "pnnx.Input" && !op->has_forward) {
      this->ReverseTopo(op);
    }
  }

  CHECK(topo_operators_.size() == operators_.size())
      << "Build wrong topo queue";
  std::reverse(topo_operators_.begin(), topo_operators_.end());

  graph_state_ = GraphState::Complete;
  input_name_ = input_name;
  output_name_ = output_name;
  if (graph_ != nullptr) {
    graph_.reset();
    graph_ = nullptr;
  }
}

std::vector<std::shared_ptr<Tensor<float>>> RuntimeGraph::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs, bool debug) {
  // 检查当前的执行图是否已经初始化完毕
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!";
  }
  CHECK(graph_state_ == GraphState::Complete)
      << "Graph status error, current state is " << int(graph_state_);

  CHECK(topo_operators_.size() == operators_.size())
      << "Build wrong topo queue";

  for (const auto& op : topo_operators_) {
    op->has_forward = false;
  }
  if (debug) {
    utils::LayerTimeStatesSingleton::LayerTimeStatesCollectorInit();
  }

  for (const auto& current_op : topo_operators_) {
    if (current_op->type == "pnnx.Input") {
      current_op->has_forward = true;
      ProbeNextLayer(current_op, inputs);
    } else if (current_op->type == "pnnx.Output") {
      current_op->has_forward = true;
      CHECK(current_op->input_operands_seq.size() == 1);
      current_op->output_operands = current_op->input_operands_seq.front();
    } else {
      InferStatus status;
      if (debug) {
        {
          utils::LayerTimeLogging layer_time_logging(current_op->name,
                                                     current_op->type);
          status = current_op->layer->Forward();
        }
      } else {
        status = current_op->layer->Forward();
      }
      CHECK(status == InferStatus::kInferSuccess)
          << current_op->layer->layer_name()
          << " layer forward failed, error code: " << int(status);
      current_op->has_forward = true;

      ProbeNextLayer(current_op, current_op->output_operands->datas);
    }
  }

  if (debug) {
    utils::LayerTimeLogging::SummaryLogging();
  }

  for (const auto& op : topo_operators_) {
    LOG_IF(FATAL, !op->has_forward)
        << "The operator: " << op->name << " has not been forward yet!";
  }

  if (operators_maps_.find(output_name_) != operators_maps_.end()) {
    const auto& output_op = operators_maps_.at(output_name_);
    CHECK(output_op->output_operands != nullptr)
        << "Output from" << output_op->name << " is empty";
    const auto& output_operand = output_op->output_operands;
    return output_operand->datas;
  } else {
    LOG(FATAL) << "Can not find the output operator " << output_name_;
    return std::vector<std::shared_ptr<Tensor<float>>>{};
  }
}

std::shared_ptr<Layer> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  auto layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed " << op->type;
  return layer;
}

void RuntimeGraph::InitGraphOperatorsInput(
    const std::vector<pnnx::Operand*>& inputs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const pnnx::Operand* input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator* producer = input->producer;
    std::shared_ptr<RuntimeOperand> runtime_operand =
        std::make_shared<RuntimeOperand>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = input->shape;

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

void RuntimeGraph::InitGraphParams(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const auto& [name, parameter] : params) {
    const int type = parameter.type;
    switch (type) {
      case int(RuntimeParameterType::kParameterUnknown): {
        std::shared_ptr<RuntimeParameter> runtime_parameter =
            std::make_shared<RuntimeParameter>();
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterBool): {
        std::shared_ptr<RuntimeParameterBool> runtime_parameter =
            std::make_shared<RuntimeParameterBool>();
        runtime_parameter->value = parameter.b;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterInt): {
        std::shared_ptr<RuntimeParameterInt> runtime_parameter =
            std::make_shared<RuntimeParameterInt>();
        runtime_parameter->value = parameter.i;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloat): {
        std::shared_ptr<RuntimeParameterFloat> runtime_parameter =
            std::make_shared<RuntimeParameterFloat>();
        runtime_parameter->value = parameter.f;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterString): {
        std::shared_ptr<RuntimeParameterString> runtime_parameter =
            std::make_shared<RuntimeParameterString>();
        runtime_parameter->value = parameter.s;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterIntArray): {
        std::shared_ptr<RuntimeParameterIntArray> runtime_parameter =
            std::make_shared<RuntimeParameterIntArray>();
        runtime_parameter->value = parameter.ai;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloatArray): {
        std::shared_ptr<RuntimeParameterFloatArray> runtime_parameter =
            std::make_shared<RuntimeParameterFloatArray>();
        runtime_parameter->value = parameter.af;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int(RuntimeParameterType::kParameterStringArray): {
        std::shared_ptr<RuntimeParameterStringArray> runtime_parameter =
            std::make_shared<RuntimeParameterStringArray>();
        runtime_parameter->value = parameter.as;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type: " << type;
      }
    }
  }
}

void RuntimeGraph::InitGraphAttrs(
    const std::map<std::string, pnnx::Attribute>& attrs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const auto& [name, attr] : attrs) {
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute =
            std::make_shared<RuntimeAttribute>();
        runtime_attribute->type = RuntimeDataType::kTypeFloat32;
        runtime_attribute->weight_data = attr.data;
        runtime_attribute->shape = attr.shape;
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown attribute type: " << attr.type;
      }
    }
  }
}

void RuntimeGraph::ProbeNextLayer(
    const std::shared_ptr<RuntimeOperator>& current_op,
    const std::vector<std::shared_ptr<Tensor<float>>>& layer_output_datas) {
  // 当前节点的后继节点next_ops
  const auto& next_ops = current_op->output_operators;
  // 对所有后继节点进行遍历
  for (const auto& [_, next_op] : next_ops) {
    // 得到后继节点的输入next_input_operands
    const auto& next_input_operands = next_op->input_operands;
    // 确定后继节点的输入来自于current_op
    const auto& next_input_operands_iter =
        next_input_operands.find(current_op->name);
    if (next_input_operands_iter != next_input_operands.end()) {
      /**
       * 得到后继节点的关于current_op输出的输入空间 next_input_datas
       * next_input_operands:
       * {
       *    输入1 -- current_op.name: current_op对应的输出空间
       *    输入2 -- other_op.name: other_op对应的输出空间
       * }
       */
      std::vector<std::shared_ptr<ftensor>>& next_input_datas =
          next_input_operands_iter->second->datas;
      CHECK(next_input_datas.size() == layer_output_datas.size())
          << "Input data size do not match with output data size";
      // 将当前current_op的输出赋值到next_input_datas中
      for (int i = 0; i < next_input_datas.size(); ++i) {
        sftensor layer_output_data = layer_output_datas.at(i);
        CHECK(layer_output_data != nullptr);
        if (next_input_datas.at(i) != nullptr) {
          CHECK(next_input_datas.at(i)->shapes() == layer_output_data->shapes())
              << "Input data shape do not match with output data shapes";
        }
        next_input_datas.at(i) = layer_output_data;
      }
    }
  }
}

void RuntimeGraph::ReverseTopo(
    const std::shared_ptr<RuntimeOperator>& root_op) {
  CHECK(root_op != nullptr) << "current operator is nullptr";
  root_op->has_forward = true;
  const auto& next_ops = root_op->output_operators;
  for (const auto& [_, op] : next_ops) {
    if (op != nullptr) {
      if (!op->has_forward) {
        this->ReverseTopo(op);
      }
    }
  }
  for (const auto& [_, op] : next_ops) {
    CHECK_EQ(op->has_forward, true);
  }
  this->topo_operators_.push_back(root_op);
}

void RuntimeGraph::ReBuildGraph(const std::string& input_name,
                                const std::string& output_name) {
  this->graph_state_ = GraphState::NeedInit;
  this->Build(input_name, output_name);
}

RuntimeGraph::GraphState RuntimeGraph::graph_state() const {
  return this->graph_state_;
}

}  // namespace kuiper_infer
