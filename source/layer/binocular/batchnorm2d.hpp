//
// Created by fss on 22-11-17.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_BATCHNORM2D_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_BATCHNORM2D_HPP_

#include "layer/abstract/param_layer.hpp"
namespace kuiper_infer {

class BatchNorm2dLayer : public ParamLayer {
 public:
  explicit BatchNorm2dLayer(uint32_t num_features, double eps);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;

  static ParseParameterAttrStatus GetInstance(const std::shared_ptr<RuntimeOperator> &op,
                                              std::shared_ptr<Layer> &batch_layer);
 private:
  uint32_t num_features_ = 0;
  double eps_ = 1e-5;
};
}

#endif //KUIPER_COURSE_SOURCE_LAYER_BATCHNORM2D_HPP_
