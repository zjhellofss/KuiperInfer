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
  layer1.Forward(inputs, outputs1);

  ASSERT_EQ(outputs1.front()->channels(), 3);
  ASSERT_EQ(outputs1.front()->rows(), 32);
  ASSERT_EQ(outputs1.front()->cols(), 32);
  const auto &output1 = outputs1.front();
  for (int i = 0; i < output1->size(); ++i) {
    ASSERT_EQ(output1->index(i), values.at(i));
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs2;
  layer2.Forward(inputs, outputs2);
  ASSERT_EQ(outputs2.front()->channels(), 32);
  ASSERT_EQ(outputs2.front()->rows(), 3);
  ASSERT_EQ(outputs2.front()->cols(), 32);
  const auto &output2 = outputs2.front();
  for (int i = 0; i < output2->size(); ++i) {
    ASSERT_EQ(output2->index(i), values.at(i));
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs3;
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
  layer6.Forward(inputs, outputs6);
  ASSERT_EQ(outputs6.front()->channels(), 3);
  ASSERT_EQ(outputs6.front()->rows(), 32);
  ASSERT_EQ(outputs6.front()->cols(), 32);
  const auto &output6 = outputs6.front();
  for (int i = 0; i < output6->size(); ++i) {
    ASSERT_EQ(output6->index(i), values.at(i));
  }
}

TEST(test_layer, view1) {
  using namespace kuiper_infer;
  ViewLayer view_layer({1, 3, 32, -1});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(2, 32, 3);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(inputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.at(0), 3);
  ASSERT_EQ(shapes.at(1), 32);
  ASSERT_EQ(shapes.at(2), 2);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.at(0), 3);
  ASSERT_EQ(raw_shapes.at(1), 32);
  ASSERT_EQ(raw_shapes.at(2), 2);
}

TEST(test_layer, view2) {
  using namespace kuiper_infer;
  ViewLayer view_layer({1, 32, 3, 3});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 32, 3);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(inputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.at(0), 32);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 3);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.at(0), 32);
  ASSERT_EQ(raw_shapes.at(1), 3);
  ASSERT_EQ(raw_shapes.at(2), 3);
}

TEST(test_layer, view3) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 96, 3, -1});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 32, 3);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.at(0), 96);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 1);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.at(0), 96);
  ASSERT_EQ(raw_shapes.at(1), 3);
  ASSERT_EQ(raw_shapes.at(2), 1);
}

TEST(test_layer, view4) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 3, 32, -1});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(3, 32, 4);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.at(0), 3);
  ASSERT_EQ(shapes.at(1), 32);
  ASSERT_EQ(shapes.at(2), 4);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.at(0), 3);
  ASSERT_EQ(raw_shapes.at(1), 32);
  ASSERT_EQ(raw_shapes.at(2), 4);
}

TEST(test_layer, view5) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 32});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 1, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.size(), 1);
  ASSERT_EQ(raw_shapes.at(0), 32);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.size(), 3);
  ASSERT_EQ(shapes.at(0), 1);
  ASSERT_EQ(shapes.at(1), 32);
  ASSERT_EQ(shapes.at(2), 1);
}

TEST(test_layer, view6) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 96});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 3, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(shapes.size(), 1);
  ASSERT_EQ(shapes.at(0), 96);

  const auto &shapes2 = outputs.front()->shapes();
  ASSERT_EQ(shapes2.size(), 3);
  ASSERT_EQ(shapes2.at(0), 1); // channels
  ASSERT_EQ(shapes2.at(1), 96); // rows
  ASSERT_EQ(shapes2.at(2), 1); // cols
}

TEST(test_layer, view7) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 2, 3, 24});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(24, 6, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(shapes.size(), 3);
  ASSERT_EQ(shapes.at(0), 2);
  ASSERT_EQ(shapes.at(1), 3);
  ASSERT_EQ(shapes.at(2), 24);

  const auto &shapes2 = outputs.front()->shapes();
  ASSERT_EQ(shapes2.size(), 3);
  ASSERT_EQ(shapes2.at(0), 2); // channels
  ASSERT_EQ(shapes2.at(1), 3); // rows
  ASSERT_EQ(shapes2.at(2), 24); // cols
}

TEST(test_layer, view8) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 2, 48});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 3, 1);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes.at(0), 2);
  ASSERT_EQ(shapes.at(1), 48);

  const auto &shapes2 = outputs.front()->shapes();
  ASSERT_EQ(shapes2.size(), 3);
  ASSERT_EQ(shapes2.at(0), 1); // channels
  ASSERT_EQ(shapes2.at(1), 2); // rows
  ASSERT_EQ(shapes2.at(2), 48); // cols
}

TEST(test_layer, view9) {
  using namespace kuiper_infer;
  ViewLayer view_layer({2, 3, 48});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 6, 24);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 2);
  ASSERT_EQ(inputs.size(), 2);

  const auto &shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes.at(0), 3);
  ASSERT_EQ(shapes.at(1), 48);

  const auto &shapes2 = outputs.front()->shapes();
  ASSERT_EQ(shapes2.size(), 3);
  ASSERT_EQ(shapes2.at(0), 1); // channels
  ASSERT_EQ(shapes2.at(1), 3); // rows
  ASSERT_EQ(shapes2.at(2), 48); // cols
}


TEST(test_layer, view10) {
  using namespace kuiper_infer;
  ViewLayer view_layer({1, 3, 32, 33});
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(32, 3, 33);
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  inputs.push_back(input);

  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  view_layer.Forward(inputs, outputs);

  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(inputs.size(), 1);

  const auto &shapes = outputs.front()->shapes();
  ASSERT_EQ(shapes.at(0), 3);
  ASSERT_EQ(shapes.at(1), 32);
  ASSERT_EQ(shapes.at(2), 33);

  const auto &raw_shapes = outputs.front()->raw_shapes();
  ASSERT_EQ(raw_shapes.at(0), 3);
  ASSERT_EQ(raw_shapes.at(1), 32);
  ASSERT_EQ(raw_shapes.at(2), 33);
}