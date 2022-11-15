//
// Created by fss on 22-11-12.
//

#ifndef KUIPER_COURSE_DATA_BLOB_HPP_
#define KUIPER_COURSE_DATA_BLOB_HPP_
#include <memory>
#include <armadillo>
namespace kuiper_infer {
class Blob {
 public:
  explicit Blob() = default;

  explicit Blob(uint32_t channels, uint32_t rows, uint32_t cols);

  Blob(const Blob &) = default;

  Blob &operator=(const Blob &) = default;

  uint32_t rows() const;

  uint32_t cols() const;

  uint32_t channels() const;

  uint32_t size() const;

  bool empty() const;

  double front() const;

  std::vector<uint32_t> shapes() const;

  arma::cube &data();

  const arma::cube &data() const;

  arma::mat &at(uint32_t channel);

  const arma::mat &at(uint32_t channel) const;

  double at(uint32_t channel, uint32_t row, uint32_t col) const;

  void Padding(const std::vector<uint32_t> &pads, double padding_value);

  void Fill(double value);

  void Fill(const std::vector<double> &values);

  void Show();

  void Flatten();

  std::shared_ptr<Blob> Clone();

 private:
  arma::cube data_;
};
}

#endif //KUIPER_COURSE_DATA_BLOB_HPP_
