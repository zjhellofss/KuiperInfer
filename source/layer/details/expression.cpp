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

// Created by fss on 22-11-18.

#include "expression.hpp"
#include <stack>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
ExpressionLayer::ExpressionLayer(std::string statement)
    : NonParamLayer("Expression"), statement_(std::move(statement)) {
  parser_ = std::make_unique<ExpressionParser>(statement_);
}

bool ExpressionLayer::TokenIsOperator(Token token) const {
  return token.token_type == TokenType::TokenAdd || token.token_type == TokenType::TokenMul;
}

StatusCode ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
                                    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the expression layer is empty";
    return StatusCode::kInferInputsEmpty;
  }

  if (outputs.empty()) {
    LOG(ERROR) << "The output tensor array in the expression layer is empty";
    return StatusCode::kInferOutputsEmpty;
  }

  CHECK(this->parser_ != nullptr) << "The parser in the expression layer is null!";
  this->parser_->Tokenizer(false);
  const auto& tokens = this->parser_->tokens();
  const auto& token_str_array = this->parser_->token_str_array();
  CHECK(!tokens.empty() && !token_str_array.empty())
      << "The expression parser failed to parse " << statement_;

  const uint32_t batch_size = outputs.size();
  std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
  for (auto iter = tokens.rbegin(); iter != tokens.rend(); ++iter) {
    const auto& current_token = *iter;
    // 如果是数据类型，就将对应分支的input插入到栈中
    if (current_token.token_type == TokenType::TokenInputNumber) {
      std::string str_number = *(token_str_array.rbegin() + std::distance(tokens.rbegin(), iter));
      str_number.erase(str_number.begin());

      int32_t input_branch = std::stoi(str_number);
      CHECK(input_branch >= 0) << "Input branch must be >= 0";
      uint32_t input_start_pos = input_branch * batch_size;
      std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
      for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(i + input_start_pos < inputs.size())
            << "The " << i << "th operand doesn't have appropriate number of tensors";
        input_token_nodes.push_back(inputs.at(i + input_start_pos));
      }
      op_stack.push(input_token_nodes);
    } else if (TokenIsOperator(current_token)) {
      // process operation
      CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
      std::shared_ptr<Tensor<float>> (*function)(const std::shared_ptr<Tensor<float>>& tensor1,
                                                 const std::shared_ptr<Tensor<float>>& tensor2);
      std::vector<std::shared_ptr<Tensor<float>>> input_node1(std::move(op_stack.top()));
      CHECK(input_node1.size() == batch_size)
          << "The first operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();
      std::vector<std::shared_ptr<Tensor<float>>> input_node2(std::move(op_stack.top()));
      CHECK(input_node2.size() == batch_size)
          << "The second operand doesn't have appropriate number of tensors, "
             "which need "
          << batch_size;
      op_stack.pop();

      std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
      if (current_token.token_type == TokenType::TokenAdd) {
        function = TensorElementAdd;
      } else if (current_token.token_type == TokenType::TokenMul) {
        function = TensorElementMultiply;
      } else {
        LOG(FATAL) << "Unsupported operator type in the expression layer: "
                   << int(current_token.token_type);
      }
#pragma omp parallel for num_threads(batch_size)
      for (uint32_t i = 0; i < batch_size; ++i) {
        output_token_nodes.at(i) = function(input_node1.at(i), input_node2.at(i));
      }
      op_stack.push(output_token_nodes);
    }
  }
  CHECK(op_stack.size() == 1) << "The expression has more than one output operand!";
  std::vector<sftensor> output_node = op_stack.top();
  for (uint32_t i = 0; i < batch_size; ++i) {
    if (outputs.at(i) != nullptr && !outputs.at(i)->empty()) {
      CHECK(outputs.at(i)->shapes() == output_node.at(i)->shapes());
    }
    outputs.at(i) = output_node.at(i);
  }
  return StatusCode::kSuccess;
}

StatusCode ExpressionLayer::CreateInstance(const std::shared_ptr<RuntimeOperator>& op,
                                           std::shared_ptr<Layer<float>>& expression_layer) {
  if (!op) {
    LOG(ERROR) << "The expression operator parameter in the layer is null pointer.";
    return StatusCode::kParseNullOperator;
  }

  const auto& params = op->params;
  if (params.empty()) {
    LOG(ERROR) << "The operator parameter in the expression layer is empty.";
    return StatusCode::kParseParameterError;
  }

  if (params.find("expr") == params.end()) {
    return StatusCode::kParseParameterError;
  }

  auto statement_param = std::dynamic_pointer_cast<RuntimeParameterString>(params.at("expr"));
  if (statement_param == nullptr) {
    LOG(ERROR) << "Can not find the expression parameter";
    return StatusCode::kParseParameterError;
  }
  if (statement_param->type != RuntimeParameterType::kParameterString) {
    LOG(ERROR) << "Can not find the expression parameter";
    return StatusCode::kParseParameterError;
  }

  expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);
  return StatusCode::kSuccess;
}

LayerRegistererWrapper kExpressionCreateInstance(ExpressionLayer::CreateInstance,
                                                 "pnnx.Expression");
}  // namespace kuiper_infer