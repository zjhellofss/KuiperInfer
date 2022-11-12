//
// Created by fss on 22-11-12.
//

#include "blob.hpp"
#include <glog/logging.h>

#include <memory>

namespace kuiper_infer {
Blob::Blob(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::cube(rows, cols, channels);
}

uint32_t Blob::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

uint32_t Blob::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

uint32_t Blob::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

uint32_t Blob::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

bool Blob::empty() const {
  return this->data_.empty();
}

double Blob::front() const {
  return this->data_.front();
}

std::vector<uint32_t> Blob::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

arma::cube &Blob::data() {
  return this->data_;
}

const arma::cube &Blob::data() const {
  return this->data_;
}

arma::mat &Blob::at(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

const arma::mat &Blob::at(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

double Blob::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

void Blob::Padding(const std::vector<uint32_t> &pads, double padding_value) {
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

void Blob::Fill(double value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

void Blob::Fill(const std::vector<double> &values, bool need_transpose) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  const uint32_t planes = this->rows() * this->cols();
  const uint32_t channels = this->data_.n_slices;

  for (uint32_t i = 0; i < channels; ++i) {
    auto &channel_data = this->data_.slice(i);
    channel_data = arma::mat(values.data() + i * planes, this->rows(), this->cols());
    if (need_transpose) {
      CHECK_EQ(this->cols(), this->rows());
      channel_data = channel_data.t();
    }
  }
}

void Blob::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << '\n' << this->data_.slice(i);
  }
}

void Blob::Flatten() {
  CHECK(!this->data_.empty());
  const uint32_t size = this->data_.size();
  arma::cube linear_cube(1, size, 1);

  uint32_t channel = this->channels();
  uint32_t rows = this->rows();
  uint32_t cols = this->cols();
  uint32_t index = 0;
  for (uint32_t c = 0; c < channel; ++c) {
    for (uint32_t r = 0; r < rows; ++r) {
      for (uint32_t c_ = 0; c_ < cols; ++c_) {
        linear_cube.at(0, index, 0) = this->data_.at(r, c_, c);
        index += 1;
      }
    }
  }
  CHECK_EQ(index, size);
  this->data_ = linear_cube;
}

std::shared_ptr<Blob> Blob::Clone() {
  return std::make_shared<Blob>(*this);
}

}
