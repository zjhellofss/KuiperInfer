//
// Created by fss on 22-12-9.
//

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "runtime/runtime_ir.hpp"
#include "../source/layer/details/view.hpp"
TEST(test_layer, forward_view) {
  using namespace kuiper_infer;
  kuiper_infer::ViewLayer layer1({1, 3, 32, 32});
  kuiper_infer::ViewLayer layer2({1, 32, 3, 32});
  kuiper_infer::ViewLayer layer3({1, 32, 32, 3});
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3 * 32 * 32);
  const uint32_t elem_size = 3 * 32 * 32;
  std::vector<float> values;
  for (uint32_t i = 0; i < elem_size; ++i) {
    values.push_back(float(i));
  }
  input->Fill(values);

  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs1;
  outputs1.push_back(std::make_shared<Tensor<float>>(3, 32, 32));
  layer1.Forward(inputs, outputs1);

  ASSERT_EQ(outputs1.front()->channels(), 3);
  ASSERT_EQ(outputs1.front()->rows(), 32);
  ASSERT_EQ(outputs1.front()->cols(), 32);
  const auto &output1 = outputs1.front();
  for (int i = 0; i < output1->size(); ++i) {
    ASSERT_EQ(output1->index(i), values.at(i));
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs2;
  outputs2.push_back(std::make_shared<Tensor<float>>(32, 3, 32));
  layer2.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.front()->channels(), 32);
  ASSERT_EQ(outputs2.front()->rows(), 3);
  ASSERT_EQ(outputs2.front()->cols(), 32);
  const auto &output2 = outputs2.front();
  for (int i = 0; i < output2->size(); ++i) {
    ASSERT_EQ(output2->index(i), values.at(i));
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs3;
  outputs3.push_back(std::make_shared<Tensor<float>>(32, 32, 3));
  layer3.Forward(inputs, outputs3);
  ASSERT_EQ(outputs3.front()->channels(), 32);
  ASSERT_EQ(outputs3.front()->rows(), 32);
  ASSERT_EQ(outputs3.front()->cols(), 3);

  const auto &output3 = outputs3.front();
  for (int i = 0; i < output3->size(); ++i) {
    ASSERT_EQ(output3->index(i), values.at(i));
  }

  kuiper_infer::ViewLayer layer4({1, 32, 32, -1});

  std::vector<std::shared_ptr<Tensor<float>>> outputs4;
  outputs4.push_back(std::make_shared<Tensor<float>>(32, 32, 3));
  layer4.Forward(inputs, outputs4);
  ASSERT_EQ(outputs4.front()->channels(), 32);
  ASSERT_EQ(outputs4.front()->rows(), 32);
  ASSERT_EQ(outputs4.front()->cols(), 3);
  const auto &output4 = outputs4.front();
  for (int i = 0; i < output4->size(); ++i) {
    ASSERT_EQ(output4->index(i), values.at(i));
  }


  kuiper_infer::ViewLayer layer5({1, 32, 3, -1});

  std::vector<std::shared_ptr<Tensor<float>>> outputs5;
  outputs5.push_back(std::make_shared<Tensor<float>>(32, 3, 32));
  layer5.Forward(inputs, outputs5);
  ASSERT_EQ(outputs5.front()->channels(), 32);
  ASSERT_EQ(outputs5.front()->rows(), 3);
  ASSERT_EQ(outputs5.front()->cols(), 32);
  const auto &output5 = outputs5.front();
  for (int i = 0; i < output5->size(); ++i) {
    ASSERT_EQ(output5->index(i), values.at(i));
  }


  kuiper_infer::ViewLayer layer6({1, 3, 32, -1});
  std::vector<std::shared_ptr<Tensor<float>>> outputs6;
  outputs6.push_back(std::make_shared<Tensor<float>>(3, 32, 32));
  layer6.Forward(inputs, outputs6);
  ASSERT_EQ(outputs6.front()->channels(), 3);
  ASSERT_EQ(outputs6.front()->rows(), 32);
  ASSERT_EQ(outputs6.front()->cols(), 32);
  const auto &output6 = outputs6.front();
  for (int i = 0; i < output6->size(); ++i) {
    ASSERT_EQ(output6->index(i), values.at(i));
  }
}