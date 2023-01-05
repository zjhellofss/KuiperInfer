//
// Created by fss on 23-1-5.
//

#ifndef KUIPER_INFER_DEMOS_IMAGE_UTIL_HPP_
#define KUIPER_INFER_DEMOS_IMAGE_UTIL_HPP_
#include<opencv2/opencv.hpp>
float Letterbox(
    const cv::Mat &image,
    cv::Mat &out_image,
    const cv::Size &new_shape = cv::Size(640, 640),
    int stride = 32,
    const cv::Scalar &color = cv::Scalar(114, 114, 114),
    bool fixed_shape = false,
    bool scale_up = false);

#endif //KUIPER_INFER_DEMOS_IMAGE_UTIL_HPP_
