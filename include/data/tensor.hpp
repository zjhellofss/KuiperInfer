//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_DATA_BLOB_HPP_
#define KUIPER_COURSE_DATA_BLOB_HPP_
#include <memory>
#include "armadillo"
namespace kuiper_infer {
class Tensor {
 public:
  explicit Tensor() = default;

  explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

  Tensor(const Tensor &) = default;

  Tensor &operator=(const Tensor &) = default;

  uint32_t rows() const;

  uint32_t cols() const;

  uint32_t channels() const;

  uint32_t size() const;

  bool empty() const;

  double index(uint32_t offset) const;

  std::vector<uint32_t> shapes() const;

  arma::cube &data();

  const arma::cube &data() const;

  arma::mat &at(uint32_t channel);

  const arma::mat &at(uint32_t channel) const;

  double at(uint32_t channel, uint32_t row, uint32_t col) const;

  double &at(uint32_t channel, uint32_t row, uint32_t col);

  void Padding(const std::vector<uint32_t> &pads, double padding_value);

  void Fill(double value);

  void Fill(const std::vector<double> &values);

  void Ones();

  void Rand();

  void Show();

  void Concat(const std::shared_ptr<Tensor> &tensor);

  void Add(const std::shared_ptr<Tensor> &tensor);

  void Flatten();

  void Transform(const std::function<double(double)> &filter);

  std::shared_ptr<Tensor> Clone();

  arma::cube data_;
};

}

#endif //KUIPER_COURSE_DATA_BLOB_HPP_
