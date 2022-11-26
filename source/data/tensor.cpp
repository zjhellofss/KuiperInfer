//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>

#include <memory>

namespace kuiper_infer {
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
}

uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

void Tensor<float>::set_data(const arma::fcube &data) {
  CHECK(data.n_rows == data.n_rows);
  CHECK(data.n_cols == data.n_cols);
  CHECK(data.n_slices == data.n_slices);
  this->data_ = data;
}

bool Tensor<float>::empty() const {
  return this->data_.empty();
}

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size());
  return this->data_.at(offset);
}

std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::fcube &Tensor<float>::data() {
  return this->data_;
}

const arma::fcube &Tensor<float>::data() const {
  return this->data_;
}

arma::fmat &Tensor<float>::at(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::fmat &Tensor<float>::at(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

float &Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Tensor<float>::Padding(const std::vector<uint32_t> &pads, float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  arma::fcube padded_cube;
  uint32_t channels = this->channels();
  CHECK_GT(channels, 0);

  for (uint32_t i = 0; i < channels; ++i) {
    const arma::fmat &sub_mat = this->data_.slice(i);
    CHECK(!sub_mat.empty());

    arma::fmat padded_mat(sub_mat.n_rows + pad_rows1 + pad_rows2,
                          sub_mat.n_cols + pad_cols1 + pad_cols2);

    padded_mat.fill((float) padding_value);
    padded_mat.submat(pad_rows1, pad_cols1, pad_rows1 + sub_mat.n_rows - 1,
                      pad_cols1 + sub_mat.n_cols - 1) = sub_mat;

    if (padded_cube.empty()) {
      padded_cube = arma::fcube(padded_mat.n_rows, padded_mat.n_cols, channels);
    }
    padded_cube.slice(i) = padded_mat;
  }
  CHECK(!padded_cube.empty());
  this->data_ = padded_cube;
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Tensor<float>::Fill(const std::vector<float> &values) {
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
    channel_data = arma::fmat(values.data() + i * planes, this->rows(), this->cols());
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        channel_data.at(r, c) = values.at(index);
        index += 1;
      }
    }
  }
}

void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

void Tensor<float>::Flatten() {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  arma::fcube linear_cube(1, size, 1);

  uint32_t channel = this->channels();
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();
  uint32_t index = 0;

  for (uint32_t c = 0; c < channel; ++c) {
    const arma::fmat &matrix = this->data_.slice(c);

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

std::shared_ptr<Tensor<float>> Tensor<float>::Clone() {
  return std::make_shared<Tensor>(*this);
}

void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->data_.fill(1.);
}

void Tensor<float>::Concat(const std::shared_ptr<Tensor<float>> &tensor) {
  CHECK(!this->data_.empty() && !tensor->data_.empty());
  const uint32_t total_channel = this->data_.n_slices + tensor->data_.n_slices;
  CHECK(this->rows() == tensor->rows() && this->cols() == tensor->cols()) << "Tensors shape are not adapting";
  this->data_ = arma::join_slices(this->data_, tensor->data_);
  CHECK(this->data_.n_slices == total_channel);
}

void Tensor<float>::Add(const std::shared_ptr<Tensor<float>> &tensor) {
  CHECK(!this->data_.empty() && !tensor->data_.empty());
  CHECK(this->shapes() == tensor->shapes()) << "Tensors shape are not adapting";
  this->data_ = this->data_ + tensor->data_;
}

void Tensor<float>::Transform(const std::function<float(float)> &filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

}
