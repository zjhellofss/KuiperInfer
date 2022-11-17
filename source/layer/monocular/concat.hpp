//
// Created by fss on 22-11-16.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_CONCAT_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_CONCAT_HPP_
#include "../abstract/layer.hpp"
namespace kuiper_infer {
class ConcatLayer : public Layer {
 public:
  explicit ConcatLayer(int dim) : dim_(dim), Layer("Concat") {

  }

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &concat_layer);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs1,
                      const std::vector<std::shared_ptr<Tensor>> &inputs2,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;

 private:
  int dim_ = 0;// concat的维度
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_CONCAT_HPP_
