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

// Created by fss on 22-11-17.

#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_op.hpp"

namespace kuiper_infer {
class LayerRegisterer {
 public:
  typedef ParseParameterAttrStatus (*Creator)(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& layer);

  typedef std::map<std::string, Creator> CreateRegistry;

  /**
   * 向注册表注册算子
   * @param layer_type 算子的类型
   * @param creator 需要注册算子的注册表
   */
  static void RegisterCreator(const std::string& layer_type,
                              const Creator& creator);

  /**
   * 通过算子参数op来初始化Layer
   * @param op 保存了初始化Layer信息的算子
   * @return 初始化后的Layer
   */
  static std::shared_ptr<Layer> CreateLayer(
      const std::shared_ptr<RuntimeOperator>& op);

  /**
   * 返回算子的注册表
   * @return 算子的注册表
   */
  static CreateRegistry& Registry();

  /**
   * 返回所有已被注册算子的类型
   * @return 注册算子的类型列表
   */
  static std::vector<std::string> layer_types();
};

class LayerRegistererWrapper {
 public:
  LayerRegistererWrapper(const std::string& layer_type,
                         const LayerRegisterer::Creator& creator) {
    LayerRegisterer::RegisterCreator(layer_type, creator);
  }
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
