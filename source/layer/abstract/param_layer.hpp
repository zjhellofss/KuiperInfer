//
// Created by fss on 22-11-13.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
#include"layer.hpp"

namespace kuiper_infer {
class ParamLayer : public Layer {
 public:
  explicit ParamLayer(const std::string &layer_name, const std::vector<std::shared_ptr<Tensor>> &weights,
                      const std::vector<std::shared_ptr<Tensor>> &bias);

  explicit ParamLayer(const std::string &layer_name);

  const std::vector<std::shared_ptr<Tensor>> &weights() const override;

  const std::vector<std::shared_ptr<Tensor>> &bias() const override;

  void set_weights(const std::vector<double> &weights) override;

  void set_bias(const std::vector<double> &bias) override;

  InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                      std::vector<std::shared_ptr<Tensor>> &outputs) override;

 protected:
  std::vector<std::shared_ptr<Tensor>> weights_;
  std::vector<std::shared_ptr<Tensor>> bias_;
};

}

#endif //KUIPER_COURSE_SOURCE_LAYER_PARAM_LAYER_HPP_
