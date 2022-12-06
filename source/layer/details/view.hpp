//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class ViewLayer : public Layer {
 public:
  explicit ViewLayer(const std::vector<int32_t> &shapes);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &view_layer);
 private:
  std::vector<int32_t> shapes_;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_FLATTEN_HPP_
