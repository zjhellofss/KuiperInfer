//
// Created by fss on 22-11-28.
//

#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
#include <glog/logging.h>
#include <vector>
#include "runtime_datatype.hpp"
#include "status_code.hpp"

namespace kuiper_infer {

/// 计算图节点的属性信息
struct RuntimeAttribute {
  std::vector<char> weight_data;  /// 节点中的权重参数
  std::vector<int> shape;         /// 节点中的形状信息
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;  /// 节点中的数据类型

  /**
   * 从节点中加载权重参数
   * @tparam T 权重类型
   * @return 权重参数数组
   */
  template <class T>  //
  std::vector<T> get(bool need_clear_weight = true);

  /**
   * 清除权重
   */
  void ClearWeight();
};

template <class T>
std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
  /// 检查节点属性中的权重类型
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::kTypeUnknown);
  std::vector<T> weights;
  switch (type) {
    case RuntimeDataType::kTypeFloat32: {  /// 加载的数据类型是float
      const bool is_float = std::is_same<T, float>::value;
      CHECK_EQ(is_float, true);
      const uint32_t float_size = sizeof(float);
      CHECK_EQ(weight_data.size() % float_size, 0);
      for (uint32_t i = 0; i < weight_data.size() / float_size; ++i) {
        float weight = *((float*)weight_data.data() + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unknown weight data type: " << int(type);
    }
  }
  if (need_clear_weight) {
    this->ClearWeight();
  }
  return weights;
}

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
