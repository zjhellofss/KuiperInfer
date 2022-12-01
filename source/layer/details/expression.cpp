//
// Created by fss on 22-11-18.
//

#include "expression.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
ExpressionLayer::ExpressionLayer(const std::string &statement)
    : Layer("Expression"), parser_(std::make_unique<ExpressionParser>(statement)) {

}

InferStatus ExpressionLayer::Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                                     std::vector<std::shared_ptr<Tensor<float>>> &outputs) {

  const uint32_t size = inputs.size();
  CHECK(size % 2 == 0);
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
  for (uint32_t i = 0; i < batch_size; ++i) {
    outputs.at(i)->Fill(0.f);
  }
  std::cout << inputs.at(0)->at(0).at(0) << std::endl;
  std::cout << inputs.at(4)->at(0).at(0) << std::endl;
  if (inputs.size() >= 9) {
    std::cout << inputs.at(8)->at(0).at(0) << std::endl;
  }
  if (expression.token_type == TokenType::TokenAdd) {
//#pragma omp parallel for num_threads(batch_size)
    for (uint32_t i = 0; i < batch_size; ++i) {
      inputs.at(i)->ElementAdd(inputs.at(i + batch_size));
      outputs.at(i) = inputs.at(i);
    }
    std::cout << outputs.at(0)->at(0).at(0) << std::endl;
    std::cout << "-----------\n";
  } else if (expression.token_type == TokenType::TokenMul) {
//#pragma omp parallel for num_threads(batch_size)
    for (uint32_t i = 0; i < batch_size; ++i) {
      inputs.at(i)->ElementMultiply(inputs.at(i + batch_size));
      outputs.at(i) = inputs.at(i);
    }
  } else {
    return InferStatus::kInferFailedOperationUnknown;
  }

  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus ExpressionLayer::GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                                      std::shared_ptr<Layer> &expression_layer) {
  CHECK(op != nullptr) << "ElementAdd operator is empty";
  const auto &params = op->params;
  if (params.find("expr") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  const auto &statement_param = dynamic_cast<RuntimeParameterString *>(params.at("expr"));
  if (statement_param == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }
  if (statement_param->type != RuntimeParameterType::kParameterString) {
    return ParseParameterAttrStatus::kParameterMissingExpr;
  }

  expression_layer = std::make_shared<ExpressionLayer>(statement_param->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kExpressinnGetInstance("pnnx.Expression", ExpressionLayer::GetInstance);
}
