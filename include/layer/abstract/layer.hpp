//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
#include <string>
#include <vector>
#include <memory>
#include <glog/logging.h>

#include "common.hpp"
#include "data/tensor.hpp"

namespace kuiper_infer {
class RuntimeOperator;

class Layer {
 public:
  explicit Layer(const std::string &layer_name) : layer_name_(layer_name) {

  }

  virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs,
                              std::vector<std::shared_ptr<Tensor>> &outputs);

  virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor>> &inputs1,
                              const std::vector<std::shared_ptr<Tensor>> &inputs2,
                              std::vector<std::shared_ptr<Tensor>> &ouputs);

  virtual const std::vector<std::shared_ptr<Tensor>> &weights() const;

  virtual const std::vector<std::shared_ptr<Tensor>> &bias() const;

  virtual void set_weights(const std::vector<std::shared_ptr<Tensor>> &weights);

  virtual void set_bias(const std::vector<std::shared_ptr<Tensor>> &bias);

  virtual void set_weights(const std::vector<double> &weights);

  virtual void set_bias(const std::vector<double> &bias);

  virtual const std::string &layer_name() const { return this->layer_name_; }

 protected:
  std::string layer_name_;
};

}
#endif //KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
