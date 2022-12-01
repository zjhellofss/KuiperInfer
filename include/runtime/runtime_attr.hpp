//
// Created by fss on 22-11-28.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#include <vector>
#include <glog/logging.h>
#include "status_code.hpp"

namespace kuiper_infer {
struct RuntimeAttribute {
  // ir中的层权重参数
  std::vector<char> weight_data;
  std::vector<int> shape;
  RuntimeDataType type = RuntimeDataType::kTypeNull;

  template<class T>
  std::vector<T> get();
};

template<class T>
std::vector<T> RuntimeAttribute::get() {
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::kTypeNull);
  std::vector<T> weights;
  switch (type) {
    case RuntimeDataType::kTypeFloat32: {
      const bool is_float = std::is_same<T, float>::value;
      CHECK_EQ(is_float, true);
      const uint32_t float_size = sizeof(float);
      CHECK_EQ(weight_data.size() % float_size, 0);
      for (int i = 0; i < weight_data.size() / float_size; ++i) {
        float v = *((float *) weight_data.data() + i);
        weights.push_back(v);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unknown weight data type";
    }
  }
  return weights;
}
}
#endif //KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
