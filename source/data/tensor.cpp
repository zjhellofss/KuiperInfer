//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <functional>
#include <memory>
#include <numeric>

namespace kuiper_infer {

Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  this->size_ = channels * rows * cols;

  cudaMalloc(&this->gpu_data_, sizeof(float) * this->size_);
  // cudaMemset();
  this->shapes_ = std::vector<uint32_t>{1, channels, rows, cols};
}

Tensor<float>::Tensor(uint32_t nums, uint32_t channels, uint32_t rows,
                      uint32_t cols) {
  this->size_ = nums * channels * rows * cols;

  cudaMalloc(&this->gpu_data_, sizeof(float) * this->size_);
  // cudaMemset();
  this->shapes_ = std::vector<uint32_t>{nums, channels, rows, cols};
}

Tensor<float>::~Tensor() {
  // if()
  cudaFree(this->gpu_data_);
}

Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(shapes.size() == 4);
  uint32_t channels = shapes.at(0);
  uint32_t rows = shapes.at(1);
  uint32_t cols = shapes.at(2);

  if (channels == 1 && rows == 1) {
    this->shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }

  this->size_ =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<int>());

  cudaMalloc(&this->gpu_data_, sizeof(float) * this->size_);
  this->shapes_ = shapes;
}

Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->gpu_data_ = tensor.gpu_data_;
    this->shapes_ = tensor.shapes_;
  }
}

Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->gpu_data_ = tensor.gpu_data_;
    this->shapes_ = tensor.shapes_;
  }
  return *this;
}

uint32_t Tensor<float>::rows() const {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_[2];
}

uint32_t Tensor<float>::cols() const {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_[3];
}

uint32_t Tensor<float>::channels() const {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_[1];
}

uint32_t Tensor<float>::nums() const {
  CHECK(!this->empty());
  return this->shapes_[0];
}

// void Tensor<float>::set_data(const arma::fcube& data) {
//   CHECK(data.n_rows == this->gpu_data_.n_rows)
//       << data.n_rows << " != " << this->gpu_data_.n_rows;
//   CHECK(data.n_cols == this->gpu_data_.n_cols)
//       << data.n_cols << " != " << this->gpu_data_.n_cols;
//   CHECK(data.n_slices == this->gpu_data_.n_slices)
//       << data.n_slices << " != " << this->gpu_data_.n_slices;
//   this->gpu_data_ = data;
// }

bool Tensor<float>::empty() const { return this->gpu_data_ != nullptr; }


float Tensor<float>::index(uint32_t nums, uint32_t channels, uint32_t rows,
                           uint32_t cols) {
  uint32_t offset = nums * shapes_[1] * shapes_[2] * shapes_[3] +
                    channels * shapes_[2] * shapes_[3] + rows * shapes_[3] +
                    cols;

  CHECK(offset < this->size_) << "Tensor index out of bound!";

  return this->gpu_data_[offset];
}

std::vector<uint32_t> Tensor<float>::shapes() {
  //  CHECK(!this->gpu_data_.empty());
  return this->shapes_;
}

float* Tensor<float>::data() { return this->gpu_data_; }

// float Tensor<float>::at(uint32_t nums, int32_t channel, uint32_t row,
//                         uint32_t col) const {
//   CHECK_LT(row, this->rows());
//   CHECK_LT(col, this->cols());
//   CHECK_LT(channel, this->channels());

//   return this->gpu_data_.at(row, col, channel);
// }

// float& Tensor<float>::at(uint32_t num,uint32_t channel, uint32_t row,
// uint32_t col) {
//   CHECK_LT(row, this->rows());
//   CHECK_LT(col, this->cols());
//   CHECK_LT(channel, this->channels());

//   return this->gpu_data_.at(row, col, channel);
// }

void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  //  CHECK(!this->gpu_data_.empty());
  //  CHECK_EQ(pads.size(), 4);
  //  uint32_t pad_rows1 = pads.at(0);  // up
  //  uint32_t pad_rows2 = pads.at(1);  // bottom
  //  uint32_t pad_cols1 = pads.at(2);  // left
  //  uint32_t pad_cols2 = pads.at(3);  // right
  //
  //  arma::fcube new_data(this->gpu_data_.n_rows + pad_rows1 + pad_rows2,
  //                       this->gpu_data_.n_cols + pad_cols1 + pad_cols2,
  //                       this->gpu_data_.n_slices);
  //  new_data.fill(padding_value);
  //
  //  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
  //                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) =
  //      this->gpu_data_;
  //  this->gpu_data_ = std::move(new_data);
}

void Tensor<float>::Fill(float value) {
  CHECK(!this->empty());

  element_wise_fill(this->size_, this->gpu_data_, value);
}

void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->empty());
  const uint32_t total_elems = this->size();
  CHECK_EQ(values.size(), total_elems);

  const uint32_t rows = this->rows();
  const uint32_t cols = this->cols();
  const uint32_t planes = rows * cols;
  const uint32_t channels = this->channels();

  // 在CUDA设备上分配内存
  cudaMalloc((void**)&this->gpu_data_, values.size() * sizeof(float));
  // 将数据从主机复制到设备
  cudaMemcpy(this->gpu_data_, values.data(), values.size() * sizeof(float),
             cudaMemcpyHostToDevice);
}

void Tensor<float>::Show() {
  //  for (uint32_t i = 0; i < this->channels(); ++i) {
  //    LOG(INFO) << "Channel: " << i;
  //    LOG(INFO) << "\n" << this->gpu_data_.slice(i);
  //  }void
}

// void Tensor<float>::Flatten(bool row_major) {
//   CHECK(!this->empty());
//   const uint32_t size = this->gpu_data_.size();
//   this->Reshape({size}, row_major);
// }

// void Tensor<float>::Rand() {
//   CHECK(!this->empty());
//   this->gpu_data_.randn();
// }

void Tensor<float>::Ones() {
  CHECK(!this->empty());
  this->Fill(1.f);
}

// void Tensor<float>::Transform(const std::function<float(float)>& filter) {
//   CHECK(!this->gpu_data_.empty());
//   this->gpu_data_.transform(filter);
// }

const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->shapes_.empty());
  CHECK_LE(this->shapes_.size(), 3);
  CHECK_GE(this->shapes_.size(), 1);
  return this->shapes_;
}

void Tensor<float>::slice_fill(uint32_t channel, float val) {
  // CHECK_LT(channel, this->channels());
  // return this->data_.slice(channel);
}

void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->empty());
  CHECK(!shapes.empty());
  const uint32_t origin_size = this->size();
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies<uint32_t>());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);

  this->shapes_ = shapes;
}

// std::vector<float> Tensor<float>::values(bool row_major) {
//   CHECK_EQ(this->gpu_data_.empty(), false);
//   std::vector<float> values(this->gpu_data_.size());

//   if (!row_major) {
//     std::copy(this->gpu_data_.mem, this->gpu_data_.mem +
//     this->gpu_data_.size(),
//               values.begin());
//   } else {
//     uint32_t index = 0;
//     for (uint32_t c = 0; c < this->gpu_data_.n_slices; ++c) {
//       const arma::fmat& channel = this->gpu_data_.slice(c).t();
//       std::copy(channel.begin(), channel.end(), values.begin() + index);
//       index += channel.size();
//     }
//     CHECK_EQ(index, values.size());
//   }
//   return values;
// }

float* Tensor<float>::gpu_data() { return this->gpu_data_; }

uint32_t Tensor<float>::size() { return this->size_; }

}  // namespace kuiper_infer

// namespace kuiper_infer
