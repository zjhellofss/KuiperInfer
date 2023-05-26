//
// Created by fss on 22-11-18.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#define KUIPER_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#include "layer/abstract/non_param_layer.hpp"
#include "parser/parse_expression.hpp"

namespace kuiper_infer {
class ExpressionLayer : public NonParamLayer {
 public:
  explicit ExpressionLayer(const std::string& statement);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& expression_layer);

 private:
  std::string statement_;
  std::unique_ptr<ExpressionParser> parser_;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
