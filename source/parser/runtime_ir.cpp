
#include "parser/runtime_ir.hpp"
#include "../layer/convolution.hpp"

#include <memory>
#include <queue>

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

bool RuntimeGraph::Init() {
  if (this->bin_path_.empty() || this->param_path_.empty()) {
    LOG(ERROR) << "The bin path or param path is empty";
    return false;
  }
  this->graph_ = std::make_unique<pnnx::Graph>();
  int load_result = this->graph_->load(param_path_, bin_path_);
  if (load_result != 0) {
    return false;
  }
  std::vector<pnnx::Operator *> operators = this->graph_->ops;
  if (operators.empty()) {
    LOG(ERROR) << "Do not read the layers' define";
    return false;
  }

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

      // 记录输出operand中的名称
      const std::vector<pnnx::Operand *> &outputs = op->outputs;
      for (const pnnx::Operand *output : outputs) {
        if (!output) {
          continue;
        }
        const auto &consumers = output->consumers;
        for (const auto &c : consumers) {
          runtime_operator->output_names.push_back(c->name);
        }
      }

      // 初始化算子中的attribute(权重)
      const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
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

      // 初始化算子中的parameter
      const std::map<std::string, pnnx::Parameter> &params = op->params;
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
      this->operators_.push_back(runtime_operator);
    }
  }

  //构建图关系
  const uint32_t op_sizes = this->operators_.size();
  for (uint32_t i = 0; i < op_sizes; ++i) {
    const auto &current_op = this->operators_.at(i);
    const std::vector<std::string> output_names = current_op->output_names;
    for (uint32_t j = i; j < op_sizes; ++j) {
      const auto &next_op = this->operators_.at(j);
      if (std::find(output_names.begin(), output_names.end(), next_op->name) != output_names.end()) {
        current_op->output_operators.insert({next_op->name, next_op});
      }
    }
  }
  return true;
}

void RuntimeGraph::Build() {
  LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty!";
  for (const auto &kOperator : this->operators_) {
    if (kOperator->type == "pnnx.Input") {
      this->input_operators_.push_back(kOperator);
    } else if (kOperator->type == "pnnx.Output") {
      this->output_operators_.push_back(kOperator);
    } else if (kOperator->type == "nn.Conv2d") {
      std::shared_ptr<Layer> conv_layer = RuntimeGraph::CreateLayer(kOperator->type, kOperator);
      if (conv_layer) {
        kOperator->layer = conv_layer;
      }
    }
  }
}

void RuntimeGraph::Forward(std::vector<std::shared_ptr<Blob>> input_data) {
  LOG_IF(FATAL, input_data.empty()) << "Inputs is empty";
  LOG_IF(FATAL, this->input_operators_.size() != 1) << "Only support one input one graph yet!";
  LOG_IF(FATAL, this->output_operators_.size() != 1) << "Only support one output one graph yet!";

  const auto &input_op = input_operators_.back();
  auto &output_op = output_operators_.back();
  std::queue<std::shared_ptr<RuntimeOperator>> ops_queue;
  ops_queue.push(input_op);
  bool is_first_layer = true;

  while (!ops_queue.empty()) {
    const auto &current_op = ops_queue.front();
    ops_queue.pop();
    if (!current_op) {
      break;
    }

    std::vector<std::shared_ptr<Blob>> output_data;
    if (is_first_layer) {
      output_data = input_data;
    } else {
      const std::string &current_op_name = current_op->name;
      LOG(INFO) << current_op_name;
      const auto &input_operands = current_op->input_operands;
      const uint32_t input_operands_size = input_operands.size();
      output_data = input_data;
      if (input_operands_size == 1) {

      } else if (input_operands_size == 2) {

      } else {
        LOG(FATAL) << "Unknown input size, three input";
      }
    }
    const auto &next_ops = current_op->output_operators;
    for (const auto &pair : next_ops) {
      auto &next_op = pair.second;
      auto &next_input_operands = next_op->input_operands;
      if (next_input_operands.find(current_op->name) != next_input_operands.end()) {
        next_input_operands.at(current_op->name)->datas = output_data;
        if (!next_op->has_transfer) {
          ops_queue.push(next_op);
          next_op->has_transfer = true;
        }
      }
    }

    if (is_first_layer) {
      is_first_layer = false;
    }
  }
}

std::shared_ptr<Layer> RuntimeGraph::CreateLayer(const std::string &layer_type,
                                                 const std::shared_ptr<RuntimeOperator> &op) {
  LOG_IF(FATAL, !op) << "Operator is empty!";
  if (layer_type == "nn.Conv2d") {
    std::shared_ptr<Layer> conv_layer;
    const ParseParameterAttrStatus &result = ConvolutionLayer::GetInstance(op, conv_layer);
    LOG_IF(FATAL, result != ParseParameterAttrStatus::kParameterParseSuccess)
            << "Build convolution layer failed, error code: " << int(result);
    return conv_layer;
  }
}
}