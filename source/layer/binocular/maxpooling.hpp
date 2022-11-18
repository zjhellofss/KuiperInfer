//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class MaxPoolingLayer : public Layer {
 public:
  MaxPoolingLayer(uint32_t pooling_size_h, uint32_t pooling_size_w,
                  uint32_t stride_h, uint32_t stride_w);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;
 private:
  uint32_t pooling_size_h_ = 0;
  uint32_t pooling_size_w_ = 0;
  uint32_t stride_h_ = 1;
  uint32_t stride_w_ = 1;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
