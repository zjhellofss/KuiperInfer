
#include "parser/runtime_ir.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <memory>
#include <queue>
#include <deque>

namespace kuiper_infer {

RuntimeOperator::~RuntimeOperator() {
  for (const auto &param : this->params) {
    if (param.second != nullptr) {
      delete param.second;
    }
  }
}

RuntimeGraph::RuntimeGraph(const std::string &param_path, const std::string &bin_path)
    : param_path_(param_path), bin_path_(bin_path) {

}

void RuntimeGraph::set_bin_path(const std::string &bin_path) {
  this->bin_path_ = bin_path;
}

void RuntimeGraph::set_param_path(const std::string &param_path) {
  this->param_path_ = param_path;
}

const std::string &RuntimeGraph::param_path() const {
  return this->param_path_;
}

const std::string &RuntimeGraph::bin_path() const {
  return this->bin_path_;
}

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }

  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    LOG(ERROR) << "Load param path and bin path error";
    return false;
  }

  std::vector<pnnx::Operator *> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Can not read the layers' define";
    return false;
  }

  this->operators_.clear();
  for (const pnnx::Operator *op : operators) {
    if (!op) {
      LOG(ERROR) << "Meet the empty node";
      continue;
    } else {
      std::shared_ptr<RuntimeOperator> runtime_operator = std::make_shared<RuntimeOperator>();
      // 初始化算子的名称
      runtime_operator->name = op->name;
      runtime_operator->type = op->type;

      // 初始化算子中的input
      const std::vector<pnnx::Operand *> &inputs = op->inputs;
      if (!inputs.empty()) {
        InitInputOperators(inputs, runtime_operator);
      }

      // 记录输出operand中的名称
      const std::vector<pnnx::Operand *> &outputs = op->outputs;
      if (!outputs.empty()) {
        InitOutputOperators(outputs, runtime_operator);
      }

      // 初始化算子中的attribute(权重)
      const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
      if (!attrs.empty()) {
        InitGraphAttrs(attrs, runtime_operator);
      }

      // 初始化算子中的parameter
      const std::map<std::string, pnnx::Parameter> &params = op->params;
      if (!params.empty()) {
        InitGraphParams(params, runtime_operator);
      }
      this->operators_.push_back(runtime_operator);
    }
  }

  //构建图关系
  for (const auto &current_op : this->operators_) {
    const std::vector<std::string> output_names = current_op->output_names;
    for (const auto &next_op : this->operators_) {
      if (next_op == current_op) {
        continue;
      }
      if (std::find(output_names.begin(), output_names.end(), next_op->name) != output_names.end()) {
        current_op->output_operators.insert({next_op->name, next_op});
      }
    }
  }

  graph_state_ = GraphState::NeedBuild;
  return true;
}

void RuntimeGraph::Build(const std::string &input_name, const std::string &output_name) {
  if (graph_state_ == GraphState::NeedInit) {
    bool init_graph = Init();
    LOG_IF(FATAL, !init_graph) << "Init graph failed!";
  }
  CHECK(graph_state_ == GraphState::NeedBuild) << "Graph status error, current state is " << int(graph_state_);

  LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty, may be no init";

  this->input_operators_maps_.clear();
  this->output_operators_maps_.clear();

  for (const auto &kOperator : this->operators_) {
    if (kOperator->type == "pnnx.Input") {
      this->input_operators_maps_.insert({kOperator->name, kOperator});
    } else if (kOperator->type == "pnnx.Output") {
      this->output_operators_maps_.insert({kOperator->name, kOperator});
    } else {
      std::shared_ptr<Layer> layer = RuntimeGraph::CreateLayer(kOperator);
      CHECK(layer != nullptr) << "Layer create failed!";
      if (layer) {
        kOperator->layer = layer;
      }
    }
  }
  graph_state_ = GraphState::Complete;
  input_name_ = input_name;
  output_name_ = output_name;
}

std::vector<std::shared_ptr<Tensor>> RuntimeGraph::Forward(const std::vector<std::shared_ptr<Tensor>> &input_data,
                                                           bool debug) {
  if (graph_state_ < GraphState::Complete) {
    LOG(FATAL) << "Graph need be build!";
  }
  CHECK(graph_state_ == GraphState::Complete) << "Graph status error, current state is " << int(graph_state_);

  std::shared_ptr<RuntimeOperator> input_op;
  if (input_operators_maps_.find(input_name_) == input_operators_maps_.end()) {
    LOG(FATAL) << "Can not find the input node: " << input_name_;
  } else {
    input_op = input_operators_maps_.at(input_name_);
  }

  std::shared_ptr<RuntimeOperator> output_op;
  if (output_operators_maps_.find(output_name_) == output_operators_maps_.end()) {
    LOG(FATAL) << "Can not find the output node: " << input_name_;
  } else {
    output_op = output_operators_maps_.at(output_name_);
  }

  std::deque<std::shared_ptr<RuntimeOperator>> ops_queue;
  std::deque<std::shared_ptr<RuntimeOperator>> ready_queue;

  ops_queue.push_back(input_op);
  bool is_first_layer = true;

  while (!ops_queue.empty() || !ready_queue.empty()) {
    for (auto read_op = ready_queue.begin(); read_op != ready_queue.end();) {
      if (RuntimeGraph::CheckOperatorReady(*read_op)) {
        read_op = ready_queue.erase(read_op);
        ops_queue.push_back(*read_op);
      } else {
        ++read_op;
      }
    }

    const auto &current_op = ops_queue.front();
    ops_queue.pop_front();

    if (!current_op || current_op == output_op) {
      if (debug)
        LOG(INFO) << "Model inference end";
      break;
    }

    std::vector<std::shared_ptr<Tensor>> layer_output_datas;
    if (is_first_layer) {
      layer_output_datas = input_data;
      is_first_layer = false;
    } else {
      const std::string &current_op_name = current_op->name;
      if (debug) {
        LOG(INFO) << current_op_name;
      }
      const auto &input_operands = current_op->input_operands;
      const uint32_t input_operands_size = input_operands.size();

      std::vector<std::shared_ptr<RuntimeOperand>> input_datas;
      for (const auto &input_operand : input_operands) {
        input_datas.push_back(input_operand.second);
      }

      std::vector<std::shared_ptr<Tensor>> layer_input_datas;
      for (uint32_t i = 0; i < input_operands_size; ++i) {
        const auto &input_operand_datas = input_datas.at(i)->datas;
        if (input_operand_datas.empty()) {
          if (std::find(ready_queue.begin(), ready_queue.end(), current_op) != ready_queue.end()) {
            ready_queue.push_back(current_op);
          }
        }
        for (const auto &input_operand_data : input_operand_datas) {
          layer_input_datas.push_back(input_operand_data);
        }
      }
      InferStatus status = current_op->layer->Forward(layer_input_datas, layer_output_datas);
      CHECK(status == InferStatus::kInferSuccess)
              << current_op->layer->layer_name() << " layer forward failed, error code: " << int(status);
      current_op->has_transfer = false;
    }

    const auto &next_ops = current_op->output_operators;
    for (const auto &next_op : next_ops) {
      const auto &next_rt_operator = next_op.second;
      auto &next_input_operands = next_rt_operator->input_operands;
      if (next_input_operands.find(current_op->name) != next_input_operands.end()) {
        if (debug) {
          for (const auto &output_data : layer_output_datas) {
            output_data->Show();
          }
        }
        next_input_operands.at(current_op->name)->datas = CloneData(layer_output_datas);
        if (!next_rt_operator->has_transfer) {
          ops_queue.push_back(next_rt_operator);
        }
        next_rt_operator->has_transfer = true;
      }
    }
  }

  std::vector<std::shared_ptr<Tensor>> output_datas;
  CHECK(output_op->input_operands.size() == 1) << "The graph only support one path to the output node yet!";
  const auto &input_operand = output_op->input_operands.begin();
  const auto &output_operand = input_operand->second;
  for (const auto &output_data : output_operand->datas) {
    output_datas.push_back(output_data);
  }
  return output_datas;
}

std::shared_ptr<Layer> RuntimeGraph::CreateLayer(const std::shared_ptr<RuntimeOperator> &op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  const auto &layer = LayerRegisterer::CreateLayer(op);
  LOG_IF(FATAL, !layer) << "Layer init failed";
  return layer;
}

std::vector<std::shared_ptr<Tensor>> RuntimeGraph::CloneData(const std::vector<std::shared_ptr<Tensor>> &data) {
  std::vector<std::shared_ptr<Tensor>> output_data;
  const uint32_t size = data.size();
  output_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    output_data.at(i) = data.at(i)->Clone();
  }
  return output_data;
}

void RuntimeGraph::InitInputOperators(const std::vector<pnnx::Operand *> &inputs,
                                      const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *input : inputs) {
    if (!input) {
      continue;
    }
    const pnnx::Operator *producer = input->producer;
    std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
    runtime_operand->name = producer->name;
    runtime_operand->shapes = input->shape;

    switch (input->type) {
      case 1: {
        runtime_operand->type = RuntimeDataType::kTypeFloat32;
        break;
      }
      default: {
        LOG(FATAL) << "Unknown input operand type: " << input->type;
      }
    }
    runtime_operator->input_operands.insert({producer->name, runtime_operand});
  }
}

void RuntimeGraph::InitOutputOperators(const std::vector<pnnx::Operand *> &outputs,
                                       const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const pnnx::Operand *output : outputs) {
    if (!output) {
      continue;
    }
    const auto &consumers = output->consumers;
    for (const auto &c : consumers) {
      runtime_operator->output_names.push_back(c->name);
    }
  }
}

void RuntimeGraph::InitGraphParams(const std::map<std::string, pnnx::Parameter> &params,
                                   const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &pair : params) {
    const std::string &name = pair.first;
    const pnnx::Parameter &parameter = pair.second;
    const int type = parameter.type;
    switch (type) {
      case 0: {
        RuntimeParameter *runtime_parameter = new RuntimeParameter;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case 1: {
        RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
        runtime_parameter->value = parameter.b;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case 2: {
        RuntimeParameterInt *runtime_parameter = new RuntimeParameterInt;
        runtime_parameter->value = parameter.i;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case 3: {
        RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
        runtime_parameter->value = parameter.f;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case 4: {
        RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
        runtime_parameter->value = parameter.s;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case 5: {
        RuntimeParameterIntArray *runtime_parameter = new RuntimeParameterIntArray;
        runtime_parameter->value = parameter.ai;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }

      case 6: {
        RuntimeParameterFloatArray *runtime_parameter = new RuntimeParameterFloatArray;
        runtime_parameter->value = parameter.af;
        runtime_operator->params.insert({name, runtime_parameter});
        break;
      }
      case 7: {
        RuntimeParameterStringArray *runtime_parameter = new RuntimeParameterStringArray;
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

void RuntimeGraph::InitGraphAttrs(const std::map<std::string, pnnx::Attribute> &attrs,
                                  const std::shared_ptr<RuntimeOperator> &runtime_operator) {
  for (const auto &pair : attrs) {
    const std::string &name = pair.first;
    const pnnx::Attribute &attr = pair.second;
    switch (attr.type) {
      case 1: {
        std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
        runtime_attribute->type = RuntimeDataType::kTypeFloat32;
        runtime_attribute->weight_data = attr.data;
        runtime_attribute->shape = attr.shape;
        runtime_operator->attribute.insert({name, runtime_attribute});
        break;
      }
      default : {
        LOG(FATAL) << "Unknown attribute type";
      }
    }
  }
}

bool RuntimeGraph::CheckOperatorReady(const std::shared_ptr<RuntimeOperator> &op) {
  CHECK(op != nullptr);

  const auto &input_operands_map = op->input_operands;
  for (const auto &input_operand_iter : input_operands_map) {
    const std::shared_ptr<RuntimeOperand> &runtime_operand = input_operand_iter.second;
    if (runtime_operand) {
      if (runtime_operand->datas.empty()) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

}