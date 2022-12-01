//
// Created by fss on 22-11-18.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
#include "layer/abstract/layer.hpp"
#include "parser/parse_expression.hpp"

namespace kuiper_infer {
class ExpressionLayer : public Layer {
 public:
  explicit ExpressionLayer(const std::string &statement);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &expression_layer);

 private:
  std::unique_ptr<ExpressionParser> parser_;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_MONOCULAR_EXPRESSION_HPP_
