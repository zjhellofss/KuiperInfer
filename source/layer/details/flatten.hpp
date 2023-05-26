//
// Created by fss on 22-12-9.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class FlattenLayer : public NonParamLayer {
 public:
  explicit FlattenLayer(int start_dim, int end_dim);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& flatten_layer);

 private:
  int start_dim_ = 0;
  int end_dim_ = 0;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_FLATTEN_HPP_
