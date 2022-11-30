//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class AdaptiveAveragePoolingLayer : public Layer {
 public:
  explicit AdaptiveAveragePoolingLayer(uint32_t output_h, uint32_t output_w);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                      std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;
 private:
  uint32_t output_h_ = 0;
  uint32_t output_w_ = 0;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
