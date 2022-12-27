//
// Created by fss on 22-12-27.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_TUPLE_CONSTRUCT_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_TUPLE_CONSTRUCT_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class TupleConstructLayer : public Layer {
 public:
  explicit TupleConstructLayer();

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &tuple_layer);
};
}
#endif //KUIPER_INFER_SOURCE_LAYER_DETAILS_TUPLE_CONSTRUCT_HPP_
