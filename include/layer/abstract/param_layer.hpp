//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#include"layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer {
 public:
  explicit ParamLayer(const std::string &layer_name);

  const std::vector<std::shared_ptr<Tensor<float>>> &weights() const override;

  const std::vector<std::shared_ptr<Tensor<float>>> &bias() const override;

  void set_weights(const std::vector<float> &weights) override;

  void set_bias(const std::vector<float> &bias) override;

  void set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights) override;

  void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias) override;

 protected:
  std::vector<std::shared_ptr<Tensor<float>>> weights_;
  std::vector<std::shared_ptr<Tensor<float>>> bias_;
};

}

#endif //KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
