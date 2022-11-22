//
// Created by fss on 22-11-22.
//
#include "data/im2col.hpp"
#include "data/tensor.hpp"
#include <armadillo>
#include <glog/logging.h>

namespace kuiper_infer {
arma::mat Im2Col(const std::shared_ptr<Tensor> &input_data, const std::shared_ptr<Tensor> &kernel,
                 const uint32_t stride_h, const uint32_t stride_w,
                 const uint32_t output_h, const uint32_t output_w) {

  CHECK(input_data->channels() == kernel->channels()) << "The channel of the input data and kernel is not adapting";
  const uint32_t kernel_w = kernel->cols();
  const uint32_t kernel_h = kernel->rows();
  uint32_t output_cols = 0;


  // 计算Im2col的列数
  const uint32_t input_channels = input_data->channels();
  const uint32_t input_width = input_data->cols();
  const uint32_t input_height = input_data->rows();

  for (uint32_t c = 0; c < input_width - kernel_w + 1; c += stride_w) {
    for (uint32_t r = 0; r < input_height - kernel_h + 1; r += stride_h) {
      output_cols += 1;
    }
  }

  LOG_IF(FATAL, output_cols <= 0) << "The size of Im2Cols output cols is less than zero";
  arma::mat img_cols;
  arma::mat kernel_cols;

  for (uint32_t c = 0; c < input_channels; ++c) {
    arma::mat img_cols_c(kernel_w * kernel_h, output_cols);
    double *img_cols_c_ptr = img_cols_c.memptr();
    const arma::mat &input_channel = input_data->at(c);
    const arma::mat &kernel_channel = kernel->at(c);

    for (uint32_t h = 0; h < input_height - kernel_h + 1; h += stride_h) {
      for (uint32_t w = 0; w < input_width - kernel_w + 1; w += stride_w) {
        for (uint32_t kh = 0; kh < kernel_h; ++kh) {
          for (uint32_t kw = 0; kw < kernel_w; ++kw) {
            auto *region_ptr_col = input_channel.colptr(kw + w) + h;
            *img_cols_c_ptr = *(region_ptr_col + kh);
            img_cols_c_ptr += 1;
          }
        }
      }
    }

    if (img_cols.empty()) {
      img_cols = img_cols_c;
    } else {
      img_cols = arma::join_cols(img_cols, img_cols_c);
    }

    if (kernel_cols.empty()) {
      kernel_cols = kernel_channel.as_row();
    } else {
      kernel_cols = arma::join_rows(kernel_cols, kernel_channel.as_row());
    }
  }

  arma::mat result = kernel_cols * img_cols;
  CHECK(result.size() == output_h * output_w) << "The output size of imcols is not correct";
  result.reshape(output_w, output_h);
  return result.t();
}
}
