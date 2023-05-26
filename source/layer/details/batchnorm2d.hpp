//
// Created by fss on 22-11-17.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
#define KUIPER_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_

#include "layer/abstract/param_layer.hpp"
#include "runtime/runtime_op.hpp"

namespace kuiper_infer {

class BatchNorm2dLayer : public ParamLayer {
 public:
  explicit BatchNorm2dLayer(uint32_t num_features, float eps,
                            const std::vector<float>& affine_weight,
                            const std::vector<float>& affine_bias);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& batch_layer);

 private:
  float eps_ = 1e-5;
  std::vector<float> affine_weight_;
  std::vector<float> affine_bias_;
};
}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_BATCHNORM2D_HPP_
