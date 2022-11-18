//
// Created by fss on 22-11-18.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_MONOCULAR_ADD_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_MONOCULAR_ADD_HPP_
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {
class AddLayer :public Layer{

 public:
  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs1,
                      const std::vector<std::shared_ptr<Tensor>> &inputs2,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_MONOCULAR_ADD_HPP_
