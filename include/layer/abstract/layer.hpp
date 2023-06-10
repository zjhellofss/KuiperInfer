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
    
// Created by fss on 22-11-12.

#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
#include <glog/logging.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"
#include "status_code.hpp"

namespace kuiper_infer {
class RuntimeOperator;
class Layer {
 public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {}

  virtual ~Layer() = default;

  /**
   * Layer的执行函数
   * @param inputs 层的输入
   * @param outputs 层的输出
   * @return 执行的状态
   */
  virtual InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs);

  /**
   * Layer的执行函数
   * @param current_operator 当前的operator
   * @return 执行的状态
   */
  virtual InferStatus Forward();

  /**
   * 返回层的权重
   * @return 返回的权重
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>>& weights() const;

  /**
   * 返回层的偏移量
   * @return 返回的偏移量
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>>& bias() const;

  /**
   * 设置Layer的权重
   * @param weights 权重
   */
  virtual void set_weights(
      const std::vector<std::shared_ptr<Tensor<float>>>& weights);

  /**
   * 设置Layer的偏移量
   * @param bias 偏移量
   */
  virtual void set_bias(
      const std::vector<std::shared_ptr<Tensor<float>>>& bias);

  /**
   * 设置Layer的权重
   * @param weights 权重
   */
  virtual void set_weights(const std::vector<float>& weights);

  /**
   * 设置Layer的偏移量
   * @param bias 偏移量
   */
  virtual void set_bias(const std::vector<float>& bias);

  /**
   * 返回层的名称
   * @return 层的名称
   */
  virtual const std::string& layer_name() const { return this->layer_name_; }

  /**
   * 设置层的执行算子
   * @param runtime_operator 该层的执行算子
   */
  void set_runtime_operator(
      const std::shared_ptr<RuntimeOperator>& runtime_operator);

 protected:
  std::weak_ptr<RuntimeOperator> runtime_operator_;
  std::string layer_name_;  /// Layer的名称
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
