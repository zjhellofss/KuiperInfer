// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 23-1-5.

#ifndef KUIPER_INFER_DEMOS_IMAGE_UTIL_HPP_
#define KUIPER_INFER_DEMOS_IMAGE_UTIL_HPP_
#include <opencv2/opencv.hpp>

struct Detection {
  cv::Rect box;
  float conf = 0.f;
  int class_id = -1;
};

float Letterbox(const cv::Mat& image, cv::Mat& out_image,
                const cv::Size& new_shape = cv::Size(640, 640), int stride = 32,
                const cv::Scalar& color = cv::Scalar(114, 114, 114), bool fixed_shape = false,
                bool scale_up = false);

void ScaleCoords(const cv::Size& img_shape, cv::Rect& coords, const cv::Size& img_origin_shape);

#endif  // KUIPER_INFER_DEMOS_IMAGE_UTIL_HPP_
