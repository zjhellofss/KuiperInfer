//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_DATA_BLOB_HPP_
#define KUIPER_COURSE_DATA_BLOB_HPP_
#include <memory>
#include "armadillo"

namespace kuiper_infer {
template<typename T>
class Tensor {

};

template<>
class Tensor<uint8_t> {
  // 待实现
};

template<>
class Tensor<float> {
 public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  Tensor(const Tensor &) = default;

  Tensor &operator=(const Tensor &) = default;

  uint32_t rows() const;

  uint32_t cols() const;

  uint32_t channels() const;

  uint32_t size() const;

  void set_data(const arma::fcube &data);

  bool empty() const;

  float index(uint32_t offset) const;

  std::vector<uint32_t> shapes() const;

  arma::fcube &data();

  const arma::fcube &data() const;

  arma::fmat &at(uint32_t channel);

  const arma::fmat &at(uint32_t channel) const;

  float at(uint32_t channel, uint32_t row, uint32_t col) const;

  float &at(uint32_t channel, uint32_t row, uint32_t col);

  void Padding(const std::vector<uint32_t> &pads, float padding_value);

  void Fill(float value);

  void Fill(const std::vector<float> &values);

  void Ones();

  void Rand();

  void Show();

  void Concat(const std::shared_ptr<Tensor<float>> &tensor);

  void ElementAdd(const std::shared_ptr<Tensor<float>> &tensor);

  static std::shared_ptr<Tensor<float>> ElementAdd(const std::shared_ptr<Tensor<float>> &tensor1,
                                                   const std::shared_ptr<Tensor<float>> &tensor2);

  static std::shared_ptr<Tensor<float>> ElementMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                                                   const std::shared_ptr<Tensor<float>> &tensor2);

  void ElementAdd(float value);

  void ElementMultiply(const std::shared_ptr<Tensor<float>> &tensor);

  void Flatten();

  void Transform(const std::function<float(float)> &filter);

  std::shared_ptr<Tensor> Clone();

 private:
  arma::fcube data_;
};

}

#endif //KUIPER_COURSE_DATA_BLOB_HPP_
