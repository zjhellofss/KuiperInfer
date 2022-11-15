
#include "parser/runtime_ir.hpp"
#include "../layer/convolution.hpp"
#include "../layer/build_layer.hpp"

#include <memory>

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
      for (const pnnx::Operand *input : inputs) {
        std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
        runtime_operand->shapes = input->shape;
        runtime_operand->name = input->name;
        switch (input->type) {
          case 1: {
            runtime_operand->type = DataType::kTypeFloat32;
            break;
          }
          default: {
            LOG(FATAL) << "Unknown input operand type";
          }
        }
        runtime_operator->inputs.push_back(runtime_operand);
      }

      // 初始化算子中的output
      const std::vector<pnnx::Operand *> &outputs = op->outputs;
      for (const pnnx::Operand *output : outputs) {
        std::shared_ptr<RuntimeOperand> runtime_operand = std::make_shared<RuntimeOperand>();
        runtime_operand->shapes = output->shape;
        runtime_operand->name = output->name;
        switch (output->type) {
          case 1: {
            runtime_operand->type = DataType::kTypeFloat32;
            break;
          }
          default: {
            LOG(FATAL) << "Unknown output operand type";
          }
        }
        runtime_operator->outputs.push_back(runtime_operand);
      }

      // 初始化算子中的attribute(权重)
      const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
      for (const auto &pair : attrs) {
        const std::string &name = pair.first;
        const pnnx::Attribute &attr = pair.second;
        switch (attr.type) {
          case 1: {
            std::shared_ptr<RuntimeAttribute> runtime_attribute = std::make_shared<RuntimeAttribute>();
            runtime_attribute->type = DataType::kTypeFloat32;
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
  return true;
}

void RuntimeGraph::BuildLayers() {
  LOG_IF(FATAL, this->operators_.empty()) << "Graph operators is empty!";
  for (const auto &op : this->operators_) {
    if (op->type == "pnnx.Input") {
      this->input_operators_.push_back(op);
    } else if (op->type == "pnnx.Output") {
      this->output_operators_.push_back(op);
    } else if (op->type == "nn.Conv2d") {
      std::shared_ptr<Layer> conv_layer = BuildLayer(op->type, op);
      this->layers_.push_back(conv_layer);
    }
  }
}

void RuntimeGraph::Forward(const std::vector<std::shared_ptr<Blob>> &inputs) {
  LOG_IF(FATAL, inputs.empty()) << "Inputs is empty";
  LOG_IF(FATAL, this->layers_.empty()) << "Layers is empty";
  bool is_first_layer = true;
  for (const auto &layer : this->layers_) {
    std::vector<std::shared_ptr<Blob>> outputs;
    layer->Forward(inputs, outputs);
  }
}
}