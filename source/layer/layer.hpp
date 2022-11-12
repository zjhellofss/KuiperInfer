//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
#define KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
#include <string>
#include <vector>
#include <memory>

#include "common.hpp"
#include "../data/blob.hpp"
namespace kuiper_infer {
class Layer {
 public:
  explicit Layer(const std::string &layer_name) : layer_name_(layer_name) {

  }

  virtual InferStatus Forward(const std::vector<std::shared_ptr<Blob>> &inputs,
                              std::vector<std::shared_ptr<Blob>> &outputs) = 0;

 protected:
  std::string layer_name_;
};
}
#endif //KUIPER_COURSE_SOURCE_LAYER_LAYER_HPP_
