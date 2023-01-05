//
// Created by fss on 23-1-5.
//

#include "image_util.hpp"

float Letterbox(const cv::Mat &image,
                cv::Mat &out_image,
                const cv::Size &new_shape,
                int stride,
                const cv::Scalar &color,
                bool fixed_shape,
                bool scale_up) {
  cv::Size shape = image.size();
  float r = std::min(
      (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
  if (!scale_up) {
    r = std::min(r, 1.0f);
  }

  int new_unpad[2]{
      (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

  cv::Mat tmp;
  if (shape.width != new_unpad[0] || shape.height != new_unpad[1]) {
    cv::resize(image, tmp, cv::Size(new_unpad[0], new_unpad[1]));
  } else {
    tmp = image.clone();
  }

  float dw = new_shape.width - new_unpad[0];
  float dh = new_shape.height - new_unpad[1];

  if (!fixed_shape) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  }

  dw /= 2.0f;
  dh /= 2.0f;

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

  return 1.0f / r;
}

