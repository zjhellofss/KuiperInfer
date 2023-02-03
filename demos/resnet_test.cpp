//
// Created by fss on 23-2-2.
//
#include <algorithm>
#include <cassert>
#include <opencv2/opencv.hpp>

#include "../source/layer/details/softmax.hpp"
#include "data/tensor.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

// python ref https://pytorch.org/hub/pytorch_vision_resnet/
cv::Mat PreProcessImage(const cv::Mat &image) {
  assert(!image.empty());
  // 调整输入大小
  cv::Mat resize_image;
  cv::resize(image, resize_image, cv::Size(256, 256));

  // 调整颜色顺序

  float mean_r = 0.485f;
  float mean_g = 0.456f;
  float mean_b = 0.406f;

  float var_r = 0.229f;
  float var_g = 0.224f;
  float var_b = 0.225f;

  /**
   * 图像归一化
   */
  resize_image.convertTo(resize_image, CV_32FC3, 1. / 255.);
  cv::Mat normalize_image = resize_image.clone();

  /**
   * 图像调整均值和方差
   */
  std::transform(
      normalize_image.begin<cv::Vec3f>(), normalize_image.end<cv::Vec3f>(),
      normalize_image.begin<cv::Vec3f>(),
      [mean_r, mean_g, mean_b, var_r, var_g, var_b](const cv::Vec3f &pixel) {
        return cv::Vec3f((pixel[0] - mean_r) / var_r,
                         (pixel[1] - mean_g) / var_g,
                         (pixel[2] - mean_b) / var_b);
      });

  cv::Mat rgb_image;
  cv::cvtColor(normalize_image, rgb_image, cv::COLOR_BGR2RGB);
  return rgb_image;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("usage: ./resnet_test [image path]\n");
    exit(-1);
  }
  using namespace kuiper_infer;

  const std::string &path = argv[1];

  const uint32_t batch_size = 2;
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    cv::Mat image = cv::imread(path);
    // 图像预处理
    cv::Mat normalize_image = PreProcessImage(image);

    std::vector<cv::Mat> split_images;
    cv::split(normalize_image, split_images);
    const uint32_t input_c = 3;
    const uint32_t input_w = 256;
    const uint32_t input_h = 256;
    assert(split_images.size() == input_c);

    sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);
    int index = 0;
    int offset = 0;
    // rgbrgb --> rrrgggbbb
    for (const auto &split_image : split_images) {
      assert(split_image.total() == input_w * input_h);
      const cv::Mat &split_image_t = split_image.t();
      memcpy(input->at(index).memptr(), split_image_t.data,
             sizeof(float) * split_image.total());
      index += 1;
      offset += split_image.total();
    }
    inputs.push_back(input);
  }

  printf("input is same:%d\n", is_same_tensor(inputs.at(0), inputs.at(1)));

  const std::string &param_path = "tmp/resnet/demo/resnet18_hub.pnnx.param";
  const std::string &weight_path = "tmp/resnet/demo/resnet18_hub.pnnx.bin";
  RuntimeGraph graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");

  // 推理
  TICK(forward)
  const std::vector<sftensor> outputs = graph.Forward(inputs, true);
  TOCK(forward)
  assert(outputs.size() == batch_size);

  // softmax
  std::vector<sftensor> outputs_softmax(batch_size);
  SoftmaxLayer softmax_layer;
  softmax_layer.Forward(outputs, outputs_softmax);
  assert(outputs_softmax.size() == batch_size);

  for (int i = 0; i < outputs_softmax.size(); ++i) {
    const sftensor &output_tensor = outputs_softmax.at(i);
    LOG(INFO) << output_tensor;
    assert(output_tensor->size() == 1 * 1000);
    // 找到类别概率最大的种类
    float max_prob = -1;
    int max_index = -1;
    for (int j = 0; j < output_tensor->size(); ++j) {
      float prob = output_tensor->index(j);
      if (max_prob <= prob) {
        max_prob = prob;
        max_index = j;
      }
    }
    printf("class with max prob is %f index %d\n", max_prob, max_index);
  }
  return 0;
}