// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
    
// Created by fss on 22-11-28.

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
  std::vector<int32_t> shape;         /// 节点中的形状信息
  RuntimeDataType type = RuntimeDataType::kTypeUnknown;  /// 节点中的数据类型

  /**
   * 从节点中加载权重参数
   * @tparam T 权重类型
   * @return 权重参数数组
   */
  template <class T>  //
  std::vector<T> get(bool need_clear_weight = true);
};

template <class T>
std::vector<T> RuntimeAttribute::get(bool need_clear_weight) {
  /// 检查节点属性中的权重类型
  CHECK(!weight_data.empty());
  CHECK(type != RuntimeDataType::kTypeUnknown);
  const uint32_t elem_size = sizeof(T);
  CHECK_EQ(weight_data.size() % elem_size, 0);

  std::vector<T> weights;
  switch (type) {
    case RuntimeDataType::kTypeFloat32: {  /// 加载的数据类型是float
      static_assert(std::is_same<T, float>::value == true);
      float* weight_data_ptr = reinterpret_cast<float*>(weight_data.data());
      const uint32_t weight_data_size = weight_data.size() / elem_size;
      for (uint32_t i = 0; i < weight_data_size; ++i) {
        float weight = *(weight_data_ptr + i);
        weights.push_back(weight);
      }
      break;
    }
    default: {
      LOG(FATAL) << "Unknown weight data type: " << int32_t(type);
    }
  }
  if (need_clear_weight) {
    std::vector<char> empty_vec = std::vector<char>();
    this->weight_data.swap(empty_vec);
  }
  return weights;
}

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_ATTR_HPP_
