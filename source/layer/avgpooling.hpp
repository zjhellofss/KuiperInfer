//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
#include "abstract/layer.hpp"
namespace kuiper_infer {
class AveragePoolingLayer : public Layer {
 public:
  explicit AveragePoolingLayer(uint32_t kernel_size, uint32_t padding = 0, uint32_t stride = 1);

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs)  override;
 private:
  uint32_t pooling_size_ = 0;
  uint32_t padding_ = 0;
  uint32_t stride_ = 1;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_AVGPOOLING_HPP_
