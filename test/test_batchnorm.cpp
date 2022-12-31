//
// Created by fss on 22-11-30.
//
#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "../source/layer/details/batchnorm2d.hpp"

TEST(test_layer, forward_batchnorm) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 224, 224);
  input->Rand();
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs(1);
  BatchNorm2dLayer layer(3, 0.f, {1, 1, 1}, {0, 0, 0});
  layer.set_weights(std::vector<float>{0, 0, 0});
  layer.set_bias(std::vector<float>{1, 1, 1});

  const auto status = layer.Forward(inputs, outputs);
  ASSERT_EQ(status, InferStatus::kInferSuccess);

  for (const auto &output : outputs) {
    const uint32_t size = output->size();
    float mean = 0.f;
    float var = 0.f;
    for (uint32_t i = 0; i < size; ++i) {
      mean += output->index(i);
      var += output->index(i) * output->index(i);
    }
    ASSERT_NEAR(mean / size, 0.f, 0.01f);
    ASSERT_NEAR(var / size, 1.f, 0.01f);
  }

}

