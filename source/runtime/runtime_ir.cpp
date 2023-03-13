#include "runtime/runtime_ir.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "layer/abstract/layer_factory.hpp"
#include "tick.hpp"

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
    LOG(ERROR) << "Load param path and bin path error: " << param_path_ << " "
               << bin_path_;
    return false;
  }

  std::vector<pnnx::Operator*> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
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
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::Build(const std::string& input_name,
                         const std::string& output_name) {
  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph) << "Init graph failed!";
  }

  CHECK(graph_state_ >= GraphState::NeedBuild)
      << "Graph status error, current state is " << int(graph_state_);
  LOG_IF(FATAL, this->operators_.empty())
      << "Graph operators is empty, may be no init";

  if (graph_state_ == GraphState::Complete) {
    return;
  }

  // 构建图关系
  for (const auto& current_op : this->operators_) {
    // 获取当前节点的所有后继节点names
    const std::vector<std::string>& output_names = current_op->output_names;
    for (const auto& next_op : this->operators_) {
      if (next_op == current_op) {
        continue;
      }
      // 如果其余节点的name符合当前节点的后继节点names，则将这个其余节点作为当前节点的后继
      if (std::find(output_names.begin(), output_names.end(), next_op->name) !=
          output_names.end()) {
        current_op->output_operators.insert({next_op->name, next_op});
      }
    }
  }

  this->input_operators_maps_.clear();
  this->output_operators_maps_.clear();
  for (const auto& kOperator : this->operators_) {
    if (kOperator->type == "pnnx.Input") {
      this->input_operators_maps_.insert({kOperator->name, kOperator});
    } else if (kOperator->type == "pnnx.Output") {
      this->output_operators_maps_.insert({kOperator->name, kOperator});
    } else {
      std::shared_ptr<Layer> layer = RuntimeGraph::CreateLayer(kOperator);
      CHECK(layer != nullptr) << "Layer create failed!";
      if (layer) {
        kOperator->layer = layer;
        layer->set_runtime_operator(kOperator);
      }
    }
  }

  // 初始化节点的输入和输出空间
  RuntimeOperatorUtils::InitOperatorInput(operators_);
  RuntimeOperatorUtils::InitOperatorOutput(graph_->ops, operators_);

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

  // 找到图中的输入算子
  std::shared_ptr<RuntimeOperator> input_op;
  if (input_operators_maps_.find(input_name_) == input_operators_maps_.end()) {
    LOG(FATAL) << "Can not find the input node: " << input_name_;
  } else {
    input_op = input_operators_maps_.at(input_name_);
  }

  // 找到图中的输出算子
  std::shared_ptr<RuntimeOperator> output_op;
  if (output_operators_maps_.find(output_name_) ==
      output_operators_maps_.end()) {
    LOG(FATAL) << "Can not find the output node: " << input_name_;
  } else {
    output_op = output_operators_maps_.at(output_name_);
  }

  // 输入和输出算子一般唯一
  // 执行队列中添加输入算子
  std::deque<std::shared_ptr<RuntimeOperator>> operator_queue;
  operator_queue.push_back(input_op);
  std::map<std::string, double> run_duration_infos;  /// 运行时间统计

  if (debug) {
    LOG(INFO) << "Batch Size:" << inputs.size();
    for (int i = 0; i < inputs.size(); ++i) {
      LOG(INFO) << "Input Rows: " << inputs.at(i)->rows()
                << " Cols: " << inputs.at(i)->cols()
                << " Channels: " << inputs.at(i)->channels();
    }
    LOG(INFO) << "Inference starting...";
    LOG(INFO) << "--------------------------------------------------"
              << "\n";
  }

  while (!operator_queue.empty()) {
    // 得到执行队列中的当前节点
    std::shared_ptr<RuntimeOperator> current_op = operator_queue.front();
    operator_queue.pop_front();

    if (!current_op || current_op == output_op) {
      if (debug) {
        LOG(INFO) << "Model Inference End";
      }
      break;
    }

    // 如果当前节点为输入节点，则将输入inputs直接拷贝到后继节点中
    if (current_op == input_op) {
      ProbeNextLayer(current_op, operator_queue, inputs);
    } else {
      // 如果当前节点是其他待执行节点，首先使用checkready检测它是否就绪
      std::string current_op_name = current_op->name;
      CHECK_EQ(CheckOperatorReady(current_op), true)
          << "Current operator " << current_op->name << " is not ready!";

      const auto& start = std::chrono::steady_clock::now();
      InferStatus status = current_op->layer->Forward();

      CHECK(status == InferStatus::kInferSuccess)
          << current_op->layer->layer_name()
          << " layer forward failed, error code: " << int(status);
      if (debug) {
        const double duration =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - start)
                .count();
        if (run_duration_infos.find(current_op->type) ==
            run_duration_infos.end()) {
          run_duration_infos.insert({current_op->type, duration});
        } else {
          run_duration_infos.at(current_op->type) += duration;
        }
      }

      const auto copy_start = std::chrono::steady_clock::now();
      // 将当前layer的计算输出current_op->output_operands->datas赋值到后继节点的输入中
      ProbeNextLayer(current_op, operator_queue,
                     current_op->output_operands->datas);
      const double duration =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::steady_clock::now() - copy_start)
              .count();
      if (debug) {
        if (run_duration_infos.find("Copy") == run_duration_infos.end()) {
          run_duration_infos.insert({"Copy", duration});
        } else {
          run_duration_infos.at("Copy") += duration;
        }
      }
    }
  }

  for (const auto& op : this->operators_) {
    op->meet_num = 0;
  }

  CHECK(output_op->input_operands.size() == 1)
      << "The graph only support one path to the output node yet!";
  // 计算图中最后一个节点的输入，等于整张图的输出
  const auto& output_op_input_operand = output_op->input_operands.begin();
  const auto& output_operand = output_op_input_operand->second;
  if (debug) {
    LOG(INFO) << "Model Running Information, Time Cost:";
    double duration_all = 0.;
    for (const auto& run_info : run_duration_infos) {
      LOG(INFO) << "OP type: " << run_info.first
                << " duration: " << run_info.second << " s";
      duration_all += run_info.second;
    }
    LOG(INFO) << "All time cost: " << duration_all << " s";
  }
  return output_operand->datas;
}

std::shared_ptr<Layer> RuntimeGraph::CreateLayer(
    const std::shared_ptr<RuntimeOperator>& op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  const auto& layer = LayerRegisterer::CreateLayer(op);
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
  for (const auto& pair : params) {
    const std::string& name = pair.first;
    const pnnx::Parameter& parameter = pair.second;
    const int type = parameter.type;
    switch (type) {
      case int(RuntimeParameterType::kParameterUnknown): {
        RuntimeParameter* runtime_parameter = new RuntimeParameter;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterBool): {
        RuntimeParameterBool* runtime_parameter = new RuntimeParameterBool;
        runtime_parameter->value = parameter.b;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterInt): {
        RuntimeParameterInt* runtime_parameter = new RuntimeParameterInt;
        runtime_parameter->value = parameter.i;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloat): {
        RuntimeParameterFloat* runtime_parameter = new RuntimeParameterFloat;
        runtime_parameter->value = parameter.f;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterString): {
        RuntimeParameterString* runtime_parameter = new RuntimeParameterString;
        runtime_parameter->value = parameter.s;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterIntArray): {
        RuntimeParameterIntArray* runtime_parameter =
            new RuntimeParameterIntArray;
        runtime_parameter->value = parameter.ai;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case int(RuntimeParameterType::kParameterFloatArray): {
        RuntimeParameterFloatArray* runtime_parameter =
            new RuntimeParameterFloatArray;
        runtime_parameter->value = parameter.af;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case int(RuntimeParameterType::kParameterStringArray): {
        RuntimeParameterStringArray* runtime_parameter =
            new RuntimeParameterStringArray;
        runtime_parameter->value = parameter.as;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      default: {
        LOG(FATAL) << "Unknown parameter type";
      }
    }
  }
}

void RuntimeGraph::InitGraphAttrs(
    const std::map<std::string, pnnx::Attribute>& attrs,
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  for (const auto& pair : attrs) {
    const std::string& name = pair.first;
    const pnnx::Attribute& attr = pair.second;
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
        LOG(FATAL) << "Unknown attribute type";
      }
    }
  }
}

bool RuntimeGraph::CheckOperatorReady(
    const std::shared_ptr<RuntimeOperator>& op) {
  CHECK(op != nullptr);
  CHECK(op->meet_num <= op->input_operands.size());
  if (op->meet_num == op->input_operands.size()) {
    return true;
  } else {
    return false;
  }
}

void RuntimeGraph::ProbeNextLayer(
    const std::shared_ptr<RuntimeOperator>& current_op,
    std::deque<std::shared_ptr<RuntimeOperator>>& operator_queue,
    const std::vector<std::shared_ptr<Tensor<float>>>& layer_output_datas) {
  // 当前节点的后继节点next_ops
  const auto& next_ops = current_op->output_operators;
  // 对所有后继节点进行遍历
  for (const auto& next_op : next_ops) {
    // 得到后继节点的输入next_input_operands
    const auto& next_rt_operator = next_op.second;
    const auto& next_input_operands = next_rt_operator->input_operands;
    // 确定后继节点的输入来自于current_op
    if (next_input_operands.find(current_op->name) !=
        next_input_operands.end()) {
      // 得到后继节点的关于current_op输出的输入空间 next_input_datas
      /**
       * next_input_operands:
       * {
       *    输入1 -- current_op.name: current_op对应的输出空间
       *    输入2 -- other_op.name: other_op对应的输出空间
       * }
       */
      std::vector<std::shared_ptr<ftensor>>& next_input_datas =
          next_input_operands.at(current_op->name)->datas;
      // 将当前current_op的输出赋值到next_input_datas中
      for (int i = 0; i < next_input_datas.size(); ++i) {
        next_input_datas.at(i) = layer_output_datas.at(i);
      }
      // 后继节点的访问次数加1
      next_rt_operator->meet_num += 1;
      if (std::find(operator_queue.begin(), operator_queue.end(),
                    next_rt_operator) == operator_queue.end()) {
        // 检测后继节点是否已经ready，有则入执行队列
        if (CheckOperatorReady(next_rt_operator)) {
          operator_queue.push_back(next_rt_operator);
        }
      }
    }
  }
}
}  // namespace kuiper_infer
