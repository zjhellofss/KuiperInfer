//
// Created by fss on 22-11-18.
//

#include "expression.hpp"
#include "layer/abstract/layer_factory.hpp"
#include <stack>

namespace kuiper_infer {
ExpressionLayer::ExpressionLayer(const std::string &statement)
    : Layer("Expression"), parser_(std::make_unique<ExpressionParser>(statement)) {

}

InferStatus ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input feature map of flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t size = inputs.size();
  if (!size) {
    LOG(ERROR) << "The input feature map of add layer is null";
    return InferStatus::kInferFailedInputEmpty;
  }
  CHECK(this->parser_ != nullptr);
  this->parser_->Tokenizer(false);
  const auto &expressions = this->parser_->tokens();
  CHECK(!expressions.empty());
  const auto &expression = expressions.front();

  const uint32_t batch_size = outputs.size();
  if (batch_size == 0) {
    LOG(ERROR) << "Output of the expression layer is empty";
    return InferStatus::kInferFailedInputOutSizeAdaptingError;
  }

  for (uint32_t i = 0; i < batch_size; ++i) {
    if (outputs.at(i) == nullptr || outputs.at(i)->empty()) {
      LOG(ERROR) << "Output of the expression layer is empty";
      return InferStatus::kInferFailedInputOutSizeAdaptingError;
    }
    outputs.at(i)->Fill(0.f);
  }

  std::stack<std::vector<std::shared_ptr<Tensor<float>>>> op_stack;
  const std::vector<std::shared_ptr<TokenNode>> &token_nodes = this->parser_->Generate();
  for (const auto &token_node : token_nodes) {
    if (token_node->num_index >= 0) {
      // operator
      uint32_t start_pos = token_node->num_index * batch_size;
      std::vector<std::shared_ptr<Tensor<float>>> input_token_nodes;
      for (uint32_t i = 0; i < batch_size; ++i) {
        CHECK(i + start_pos < inputs.size());
        input_token_nodes.push_back(inputs.at(i + start_pos));
      }
      op_stack.push(input_token_nodes);
    } else {
      //operation
      const int32_t op = token_node->num_index;
      CHECK(op_stack.size() >= 2) << "The number of operand is less than two";
      std::vector<std::shared_ptr<Tensor<float>>> input_node1 = op_stack.top();

      CHECK(input_node1.size() == batch_size);
      op_stack.pop();

      std::vector<std::shared_ptr<Tensor<float>>> input_node2 = op_stack.top();
      CHECK(input_node2.size() == batch_size);
      op_stack.pop();

      CHECK(input_node1.size() == input_node2.size());
      std::vector<std::shared_ptr<Tensor<float>>> output_token_nodes(batch_size);
#pragma omp parallel for num_threads(batch_size)
      for (uint32_t i = 0; i < batch_size; ++i) {
        if (op == -int(TokenType::TokenAdd)) {
          output_token_nodes.at(i) = Tensor<float>::ElementAdd(input_node1.at(i), input_node2.at(i));
        } else if (op == -int(TokenType::TokenMul)) {
          output_token_nodes.at(i) = Tensor<float>::ElementMultiply(input_node1.at(i), input_node2.at(i));
        } else {
          LOG(FATAL) << "Unknown operator type: " << op;
        }
      }
      op_stack.push(output_token_nodes);
    }
  }

  CHECK(op_stack.size() == 1);
  std::vector<std::shared_ptr<Tensor<float>>> output_node = op_stack.top();
  op_stack.pop();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(outputs.at(i) != nullptr && !outputs.at(i)->empty());
    outputs.at(i) = output_node.at(i);
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ExpressionLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &expression_layer) {
  CHECK(op != nullptr) << "Expression operator is nullptr";
  const auto &params = op->params;
  if (params.find("expr") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  const auto &statement_param = dynamic_cast<RuntimeParameterString *>(params.at("expr"));
  if (statement_param == nullptr) {
    LOG(ERROR) << "Can not find the expression parameter";
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }
  if (statement_param->type != RuntimeParameterType::kParameterString) {
    LOG(ERROR) << "Can not find the expression parameter";
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kExpressionGetInstance("pnnx.Expression", ExpressionLayer::GetInstance);
}
