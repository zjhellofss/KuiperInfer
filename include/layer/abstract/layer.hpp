//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <glog/logging.h>

#include "status_code.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_op.hpp"

namespace kuiper_infer {

class Layer {
 public:
  explicit Layer(std::string layer_name) : layer_name_(std::move(layer_name)) {

  }

  virtual ~Layer() = default;

  /**
   * Layer的执行函数
   * @param inputs 层的输入
   * @param outputs 层的输出
   * @return 执行的状态
   */
  virtual InferStatus Forward(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                              std::vector<std::shared_ptr<Tensor<float>>> &outputs);

  /**
   * 返回层的权重
   * @return 返回的权重
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>> &weights() const;

  /**
   * 返回层的偏移量
   * @return 返回的偏移量
   */
  virtual const std::vector<std::shared_ptr<Tensor<float>>> &bias() const;

  virtual void set_weights(const std::vector<std::shared_ptr<Tensor<float>>> &weights);

  /**
   * 设置Layer的偏移量
   * @param bias 偏移量
   */
  virtual void set_bias(const std::vector<std::shared_ptr<Tensor<float>>> &bias);

  /**
   * 设置Layer的权重
   * @param weights 权重
   */
  virtual void set_weights(const std::vector<float> &weights);

  /**
   * 设置Layer的偏移量
   * @param bias 偏移量
   */
  virtual void set_bias(const std::vector<float> &bias);

  /**
   * 返回层的名称
   * @return 层的名称
   */
  virtual const std::string &layer_name() const { return this->layer_name_; }

 protected:
  std::string layer_name_; /// Layer的名称
};

}
#endif //KUIPER_INFER_SOURCE_LAYER_LAYER_HPP_
