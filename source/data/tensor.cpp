//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>

#include <memory>

namespace kuiper_infer {
Tensor::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::cube(rows, cols, channels);
}

uint32_t Tensor::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

bool Tensor::empty() const {
  return this->data_.empty();
}

double Tensor::index(uint32_t offset) const {
  CHECK(offset < this->data_.size());
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::cube &Tensor::data() {
  return this->data_;
}

const arma::cube &Tensor::data() const {
  return this->data_;
}

arma::mat &Tensor::at(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::mat &Tensor::at(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

double Tensor::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

double &Tensor::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor::Padding(const std::vector<uint32_t> &pads, double padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  arma::cube padded_cube;
  uint32_t channels = this->channels();
  CHECK_GT(channels, 0);

  for (uint32_t i = 0; i < channels; ++i) {
    const arma::mat &sub_mat = this->data_.slice(i);
    CHECK(!sub_mat.empty());

    arma::mat padded_mat(sub_mat.n_rows + pad_rows1 + pad_rows2,
                         sub_mat.n_cols + pad_cols1 + pad_cols2);

    padded_mat.fill((double) padding_value);
    padded_mat.submat(pad_rows1, pad_cols1, pad_rows1 + sub_mat.n_rows - 1,
                      pad_cols1 + sub_mat.n_cols - 1) = sub_mat;

    if (padded_cube.empty()) {
      padded_cube = arma::cube(padded_mat.n_rows, padded_mat.n_cols, channels);
    }
    padded_cube.slice(i) = padded_mat;
  }
  CHECK(!padded_cube.empty());
  this->data_ = padded_cube;
}

void Tensor::Fill(double value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor::Fill(const std::vector<double> &values) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t planes = rows * cols;
  const uint32_t channels = this->data_.n_slices;

  uint32_t index = 0;
  for (uint32_t i = 0; i < channels; ++i) {
    auto &channel_data = this->data_.slice(i);
    channel_data = arma::mat(values.data() + i * planes, this->rows(), this->cols());
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        channel_data.at(r, c) = values.at(index);
        index += 1;
      }
    }
  }
}

void Tensor::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor::Flatten() {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  arma::cube linear_cube(1, size, 1);

  uint32_t channel = this->channels();
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();
  uint32_t index = 0;

  for (uint32_t c = 0; c < channel; ++c) {
    const arma::mat &matrix = this->data_.slice(c);

    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c_ = 0; c_ < cols; ++c_) {
        linear_cube.at(0, index, 0) = matrix.at(r, c_);
        index += 1;
      }
    }
  }
  CHECK_EQ(index, size);
  this->data_ = linear_cube;
}

std::shared_ptr<Tensor> Tensor::Clone() {
  return std::make_shared<Tensor>(*this);
}

void Tensor::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor::Ones() {
  CHECK(!this->data_.empty());
  this->data_.fill(1.);
}

void Tensor::Concat(const std::shared_ptr<Tensor> &tensor) {
  CHECK(!this->data_.empty() && !tensor->data_.empty());
  const uint32_t total_channel = this->data_.n_slices + tensor->data_.n_slices;
  CHECK(this->rows() == tensor->rows() && this->cols() == tensor->cols()) << "Tensors shape are not adapting";
  this->data_ = arma::join_slices(this->data_, tensor->data_);
  CHECK(this->data_.n_slices == total_channel);
}

void Tensor::Add(const std::shared_ptr<Tensor> &tensor) {
  CHECK(!this->data_.empty() && !tensor->data_.empty());
  CHECK(this->shapes() == tensor->shapes()) << "Tensors shape are not adapting";
  this->data_ = this->data_ + tensor->data_;
}

void Tensor::Transform(const std::function<double(double)> &filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

}
