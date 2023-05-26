//
// Created by fss on 22-12-25.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
#include "layer/abstract/non_param_layer.hpp"
namespace kuiper_infer {
class CatLayer : public NonParamLayer {
 public:
  explicit CatLayer(int dim);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& cat_layer);

 private:
  int32_t dim_ = 0;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_CAT_HPP_
