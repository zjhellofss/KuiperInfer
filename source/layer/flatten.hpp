//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
#include "abstract/layer.hpp"
namespace kuiper_infer {
class FlattenLayer : public Layer {
 public:
  explicit FlattenLayer();

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &flatten_layer);

};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
