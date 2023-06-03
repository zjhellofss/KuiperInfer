//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_DATA_BLOB_HPP_
#define KUIPER_INFER_DATA_BLOB_HPP_
#include <armadillo>
#include <memory>
#include <vector>

namespace kuiper_infer {
template <typename T = float>
class Tensor {};

template <>
class Tensor<uint8_t> {
  // 待实现
};

template <>
class Tensor<float> {
 public:
  explicit Tensor() = default;

  /**
   * 创建张量
   * @param channels 张量的通道数
   * @param rows 张量的行数
   * @param cols 张量的列数
   */
  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  /**
   * 创建一个一维向量
   * @param size 一维向量中元素的个数
   */
  explicit Tensor(uint32_t size);

  /**
   * 创建一个二维向量
   * @param rows 二维向量的高度
   * @param cols 二维向量的宽度
   */
  explicit Tensor(uint32_t rows, uint32_t cols);

  /**
   * 创建张量
   * @param shapes 张量的维度
   */
  explicit Tensor(const std::vector<uint32_t>& shapes);

  Tensor(const Tensor& tensor);

  Tensor(Tensor&& tensor) noexcept;

  Tensor<float>& operator=(Tensor&& tensor) noexcept;

  Tensor<float>& operator=(const Tensor& tensor);

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
   * 返回张量的通道数
   * @return 张量的通道数
   */
  uint32_t channels() const;

  /**
   * 返回张量中元素的数量
   * @return 张量的元素数量
   */
  uint32_t size() const;

  /**
   * 设置张量中的具体数据
   * @param data 数据
   */
  void set_data(const arma::fcube& data);

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
  float index(uint32_t offset) const;

  /**
   * 返回张量中offset位置的元素
   * @param offset 需要访问的位置
   * @return offset位置的元素
   */
  float& index(uint32_t offset);

  /**
   * 张量的尺寸大小
   * @return 张量的尺寸大小
   */
  std::vector<uint32_t> shapes() const;

  /**
   * 张量的实际尺寸大小
   * @return 张量的实际尺寸大小
   */
  const std::vector<uint32_t>& raw_shapes() const;

  /**
   * 返回张量中的数据
   * @return 张量中的数据
   */
  arma::fcube& data();

  /**
   * 返回张量中的数据
   * @return 张量中的数据
   */
  const arma::fcube& data() const;

  /**
   * 返回张量第channel通道中的数据
   * @param channel 需要返回的通道
   * @return 返回的通道
   */
  arma::fmat& slice(uint32_t channel);

  /**
   * 返回张量第channel通道中的数据
   * @param channel 需要返回的通道
   * @return 返回的通道
   */
  const arma::fmat& slice(uint32_t channel) const;

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
  void Fill(const std::vector<float>& values, bool row_major = true);

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

  /**
   * 对张量中的元素进行过滤
   * @param filter 过滤函数
   */
  void Transform(const std::function<float(float)>& filter);

  /**
   * 返回数据的原始指针
   * @return 返回数据的原始指针
   */
  float* raw_ptr();

  /**
   * 返回数据的原始指针
   * @param offset 数据指针的偏移量
   * @return 返回数据的原始指针
   */
  float* raw_ptr(uint32_t offset);

  /**
   * 返回第index个矩阵的起始地址
   * @param index 第index个矩阵
   * @return 第index个矩阵的起始地址
   */
  float* matrix_raw_ptr(uint32_t index);

 private:
  std::vector<uint32_t> raw_shapes_;  // 张量数据的实际尺寸大小
  arma::fcube data_;                  // 张量数据
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_DATA_BLOB_HPP_
