//
// Created by fss on 23-1-4.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "data/tensor.hpp"
#include "image_util.hpp"

int main() {
  using namespace kuiper_infer;
  const uint32_t input_c = 3;
  const uint32_t input_h = 640;
  const uint32_t input_w = 640;

  cv::Mat image = cv::imread("imgs/0575.jpg");
  cv::Mat out_image;
  Letterbox(image, out_image, {640, 640}, 32, {114, 114, 114}, true);

  cv::Mat rgb_image;
  cv::cvtColor(out_image, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat normalize_image;
  out_image.convertTo(normalize_image, CV_32FC3, 1.f);

  int offset = 0;
  std::vector<cv::Mat> split_images;
  cv::split(normalize_image, split_images);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(input_c, input_h, input_w);
  input->Fill(0.f);

  int index = 0;
  for (const auto &split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    memcpy(input->at(index).memptr(), split_image.data, sizeof(float) * split_image.total());
    index += 1;
    offset += split_image.total();
  }

  cv::Mat test_image1(input_h, input_w, CV_32FC1, (float *) input->RawPtr());
  cv::imwrite("demo1.jpg", test_image1);

  cv::Mat test_image2(input_h, input_w, CV_32FC1, (float *) input->RawPtr() + input_w * input_h);
  cv::imwrite("demo2.jpg", test_image2);

  cv::Mat test_image3(input_h, input_w, CV_32FC1, (float *) input->RawPtr() + 2 * input_w * input_h);
  cv::imwrite("demo3.jpg", test_image3);
  return 0;
}