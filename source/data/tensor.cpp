//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>

#include <memory>

namespace kuiper_infer {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

std::shared_ptr<Tensor<float>> Tensor<float>::Create(uint32_t channels, uint32_t rows, uint32_t cols) {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

Tensor<float>::Tensor(const std::vector<uint32_t> &shapes) {
  CHECK(shapes.size() == 3);
  uint32_t channels = shapes.at(0);
  uint32_t rows = shapes.at(1);
  uint32_t cols = shapes.at(2);

  data_ = arma::fcube(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

Tensor<float>::Tensor(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float>::Tensor(Tensor<float> &&tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

Tensor<float> &Tensor<float>::operator=(Tensor<float> &&tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

Tensor<float> &Tensor<float>::operator=(const Tensor &tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
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
  CHECK(data.n_rows == this->data_.n_rows) << data.n_rows << " != " << this->data_.n_rows;
  CHECK(data.n_cols == this->data_.n_cols) << data.n_cols << " != " << this->data_.n_cols;
  CHECK(data.n_slices == this->data_.n_slices) << data.n_slices << " != " << this->data_.n_slices;
  this->data_ = data;
}

bool Tensor<float>::empty() const {
  return this->data_.empty();
}

float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor capacity is not enough!";
  return this->data_.at(offset);
}

float &Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor capacity is not enough!";
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

  arma::fcube new_data(this->data_.n_rows + pad_rows1 + pad_rows2,
                       this->data_.n_cols + pad_cols1 + pad_cols2, this->data_.n_slices);
  new_data.fill(padding_value);

  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows1 - 1,
                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) = this->data_;
  this->data_ = std::move(new_data);
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

  for (uint32_t i = 0; i < channels; ++i) {
    auto &channel_data = this->data_.slice(i);
    const arma::fmat &channel_data_t = arma::fmat(values.data() + i * planes, this->cols(), this->rows());
    channel_data = channel_data_t.t();
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
  this->ReRawshape({size});
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

std::shared_ptr<Tensor<float>> Tensor<float>::ElementAdd(const std::shared_ptr<Tensor<float>> &tensor1,
                                                         const std::shared_ptr<Tensor<float>> &tensor2) {
  CHECK(!tensor1->empty() && !tensor2->empty());
  CHECK(tensor1->shapes() == tensor2->shapes()) << "Tensors shape are not adapting";
  std::shared_ptr<Tensor<float>>
      output_tensor = std::make_shared<Tensor<float >>(tensor1->channels(), tensor1->rows(), tensor1->cols());
  output_tensor->data_ = tensor1->data_ + tensor2->data_;
  return output_tensor;
}

std::shared_ptr<Tensor<float>> Tensor<float>::ElementMultiply(const std::shared_ptr<Tensor<float>> &tensor1,
                                                              const std::shared_ptr<Tensor<float>> &tensor2) {
  CHECK(!tensor1->empty() && !tensor2->empty());
  if (tensor1->shapes() == tensor2->shapes()) {
    std::shared_ptr<Tensor<float>> output_tensor =
        std::make_shared<Tensor<float >>(tensor1->channels(), tensor1->rows(), tensor1->cols());
    output_tensor->data_ = tensor1->data_ % tensor2->data_;
    return output_tensor;
  } else {
    CHECK(tensor1->channels() == tensor2->channels()) << "Tensors shape are not adapting";
    uint32_t channels = tensor1->channels();
    std::shared_ptr<Tensor<float>> tensor1_;
    std::shared_ptr<Tensor<float>> tensor2_;

    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      tensor1_ = tensor1;
      tensor2_ = tensor2;
    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      tensor1_ = tensor2;
      tensor2_ = tensor1;
    } else {
      LOG(FATAL) << "Tensors shape are not adapting";
    }

    const std::shared_ptr<Tensor<float>>
        input_tensor2_ = std::make_shared<Tensor<float>>(channels, tensor1_->rows(), tensor1_->cols());
    for (uint32_t c = 0; c < channels; ++c) {
      input_tensor2_->data_.slice(c).fill(tensor2_->index(c));
    }
    std::shared_ptr<Tensor<float>> output_tensor =
        std::make_shared<Tensor<float>>(input_tensor2_->rows(), input_tensor2_->cols(), input_tensor2_->channels());
    output_tensor->data_ = tensor1_->data_ % input_tensor2_->data_;
    return output_tensor;
  }
}

void Tensor<float>::Transform(const std::function<float(float)> &filter) {
  CHECK(!this->data_.empty());
  uint32_t channels = this->channels();
  for (uint32_t c = 0; c < channels; ++c) {
    this->data_.slice(c).transform(filter);
  }
}

void Tensor<float>::ReRawshape(const std::vector<uint32_t> &shapes) {
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  uint32_t current_size = 1;
  for (uint32_t s : shapes) {
    current_size *= s;
  }
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    this->data_.reshape(shapes.at(0), 1, 1);
    this->raw_shapes_ = {shapes.at(0)};
  }
}

const std::vector<uint32_t> &Tensor<float>::raw_shapes() const {
  return this->raw_shapes_;
}

void Tensor<float>::ReRawView(const std::vector<uint32_t> &shapes) {
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  uint32_t current_size = 1;
  for (uint32_t s : shapes) {
    current_size *= s;
  }
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);
  std::vector<uint32_t> target_shapes; // channel row col
  if (shapes.size() == 3) {
    target_shapes = {shapes.at(0), shapes.at(1), shapes.at(2)};
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    target_shapes = {1, shapes.at(0), shapes.at(1)};
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    target_shapes = {1, shapes.at(0), 1};
    this->raw_shapes_ = {shapes.at(0)};
  }
  this->ReView(target_shapes);
}

void Tensor<float>::ReView(const std::vector<uint32_t> &shapes) {
  const uint32_t target_channels = shapes.at(0);
  const uint32_t target_rows = shapes.at(1);
  const uint32_t target_cols = shapes.at(2);
  arma::fcube new_data(target_rows, target_cols, target_channels);

  const uint32_t plane_size = target_rows * target_cols;
  for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
    const arma::fmat &channel = this->data_.slice(c);
    for (uint32_t c_ = 0; c_ < this->data_.n_cols; ++c_) {
      const float *colptr = channel.colptr(c_);
      for (uint32_t r = 0; r < this->data_.n_rows; ++r) {
        const uint32_t pos_index = c * data_.n_rows * data_.n_cols + r * data_.n_cols + c_;
        const uint32_t ch = pos_index / plane_size;
        const uint32_t row = (pos_index - ch * plane_size) / target_cols;
        const uint32_t col = (pos_index - ch * plane_size - row * target_cols);
        new_data.at(row, col, ch) = *(colptr + r);
      }
    }
  }
  this->data_ = new_data;
}

const float *Tensor<float>::raw_ptr() const {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}
}
