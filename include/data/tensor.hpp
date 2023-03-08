//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_INFER_DATA_BLOB_HPP_
#define KUIPER_INFER_DATA_BLOB_HPP_
#include <memory>
#include <vector>
#include "armadillo"

namespace kuiper_infer {
template <typename T>
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
  void Fill(const std::vector<float>& values);

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
   * 返回一个深拷贝后的张量
   * @return 新的张量
   */
  std::shared_ptr<Tensor> Clone();

  /**
   * 返回数据的原始指针
   * @return 返回数据的原始指针
   */
  const float* raw_ptr() const;

 private:
  void ReView(const std::vector<uint32_t>& shapes);
  std::vector<uint32_t> raw_shapes_;  // 张量数据的实际尺寸大小
  arma::fcube data_;                  // 张量数据
};

using ftensor = Tensor<float>;
using sftensor = std::shared_ptr<Tensor<float>>;

std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& s1,
                                               const sftensor& s2);

std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value);

/**
 * 比较tensor的值是否相同
 * @param a 输入张量1
 * @param b 输入张量2
 * @return 比较结果
 */
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相加的结果
 */
std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 张量相加
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 矩阵点乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @param output_tensor 输出张量
 */
void TensorElementMultiply(const std::shared_ptr<Tensor<float>>& tensor1,
                           const std::shared_ptr<Tensor<float>>& tensor2,
                           const std::shared_ptr<Tensor<float>>& output_tensor);

/**
 * 张量相乘
 * @param tensor1 输入张量1
 * @param tensor2 输入张量2
 * @return 张量相乘的结果
 */
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2);

/**
 * 创建一个张量
 * @param channels 通道数量
 * @param rows 行数
 * @param cols 列数
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols);

/**
 * 创建一个张量
 * @param shapes 张量的形状
 * @return 创建后的张量
 */
std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes);

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_DATA_BLOB_HPP_
