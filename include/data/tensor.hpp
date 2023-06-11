//
// Created by fss on 22-11-12.
//

#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <vector>
#include "utils/gpu_utils.cuh"
namespace kuiper_infer {

template <typename T>
class Tensor {};

template <>
class Tensor<float> {
 public:
  explicit Tensor() { this->gpu_data_ = nullptr; };

  /**
   * 创建张量
   * @param channels 张量的通道数
   * @param rows 张量的行数
   * @param cols 张量的列数
   */

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
  /**
   * 创建张量
   * @param nums 张量的数量
   * @param channels 张量的通道数
   * @param rows 张量的行数
   * @param cols 张量的列数
   */
  explicit Tensor(uint32_t nums, uint32_t channels, uint32_t rows,
                  uint32_t cols);

  explicit Tensor(const std::vector<uint32_t>& shapes);

  Tensor(const Tensor& tensor);

  Tensor<float>& operator=(const Tensor& tensor);

  Tensor(Tensor<float>&&) = delete;  // 禁止移动构造函数
  Tensor<float>& operator=(Tensor&& tensor) = delete;  // 禁止移动构造运算符


 ~Tensor();
  /**
   * 返回图片的数量
   * @return 图片的数量问题
   */
  uint32_t nums() const;

  /**
   * 返回张量的通道数
   * @return 张量的通道数
   */
  uint32_t channels() const;

  /**
   * 返回张量的行数
   * @return 张量的行数
   */
  uint32_t rows() const;

  /**
   * 返回张量的列数
   * @return 张量的列数
   */
  uint32_t cols() const;

  /**
   * 设置张量中的具体数据
   * @param data 数据
   */
  //    void set_data(const arma::fcube& data);

  /**
   * 返回张量是否为空
   * @return 张量是否为空
   */
  bool empty() const;

  /**
   * 返回张量中offset位置的元素
   * @param offset 需要访问的位置
   * @return offset位置的元素
   */
  float index(uint32_t nums, uint32_t channels, uint32_t rows, uint32_t);

  // /**
  //  * 返回张量中offset位置的元素
  //  * @param offset 需要访问的位置
  //  * @return offset位置的元素
  //  */
  // float& index(uint32_t offset);

  /**
   * 张量的尺寸大小
   * @return 张量的尺寸大小
   */
  std::vector<uint32_t> shapes();

  /**
   * 张量的实际尺寸大小
   * @return 张量的实际尺寸大小
   */
  const std::vector<uint32_t>& raw_shapes() const;

  /**
   * 返回张量中的数据指针
   * @return 张量中的数据指针
   */
  float* data();

  /**
   * 返回特定位置的元素
   * @param channel 通道
   * @param row 行数
   * @param col 列数
   * @return 特定位置的元素
   */
  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  /**
   * 返回特定位置的元素
   * @param channel 通道
   * @param row 行数
   * @param col 列数
   * @return 特定位置的元素
   */
  float& at(uint32_t channel, uint32_t row, uint32_t col);

  /**
   * 填充张量
   * @param pads 填充张量的尺寸
   * @param padding_value 填充张量
   */
  void Padding(const std::vector<uint32_t>& pads, float padding_value);

  /**
   * 使用value值去初始化向量
   * @param value
   */
  void Fill(float value);

  /**
   * 使用values中的数据初始化张量
   * @param values 用来初始化张量的数据
   */
  void Fill(const std::vector<float>& values);

  /**
   * 返回Tensor内的所有数据
   * @param row_major 是否是行主序列的
   * @return Tensor内的所有数据
   */
  std::vector<float> values(bool row_major = true);

  /**
   * 以常量1初始化张量
   */
  void Ones();

  /**
   * 以随机值初始化张量
   */
  void Rand();

  /**
   * 打印张量
   */
  void Show();

  /**
   * 张量的实际尺寸大小的Reshape pytorch兼容
   * @param shapes 张量的实际尺寸大小
   * @param row_major 根据行主序还是列主序进行reshape
   */
  void Reshape(const std::vector<uint32_t>& shapes, bool row_major = false);

  /**
   * 展开张量
   */
  void Flatten(bool row_major = false);

  // /**
  //  * 对张量中的元素进行过滤
  //  * @param filter 过滤函数
  //  */
  // void Transform(const std::function<float(float)>& filter);
  void slice_fill(uint32_t channel, float val);
  float* gpu_data();

  // 将gpu中的数据copy到cpu中用来检验计算的正确性
  std::shared_ptr<float> to_cpu_data();
  uint32_t size();

 private:
  std::vector<uint32_t> shapes_;  // 张量数据的实际尺寸大小
  float* gpu_data_;
  float* cpu_data_;  // 张量数据
  uint32_t size_;
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

}  // namespace kuiper_infer
