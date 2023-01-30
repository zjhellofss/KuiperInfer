//
// Created by fss on 23-1-22.
//

#include <gtest/gtest.h>
#include "layer/abstract/layer_factory.hpp"

using namespace kuiper_infer;
ParseParameterAttrStatus TestCreateLayer(const std::shared_ptr<RuntimeOperator> &op,
                                         std::shared_ptr<Layer> &layer) {
  layer = std::make_shared<Layer>("test3");
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

TEST(test_layer_factory, init) {
  using namespace kuiper_infer;
  const auto &r1 = LayerRegisterer::Registry();
  const auto &r2 = LayerRegisterer::Registry();
  ASSERT_EQ(r1, r2);
}

TEST(test_layer_factory, registerer) {
  using namespace kuiper_infer;
  const uint32_t size1 = LayerRegisterer::Registry().size();
  LayerRegisterer::RegisterCreator("test1", TestCreateLayer);
  const uint32_t size2 = LayerRegisterer::Registry().size();
  LayerRegisterer::RegisterCreator("test2", TestCreateLayer);
  const uint32_t size3 = LayerRegisterer::Registry().size();

  ASSERT_EQ(size2 - size1, 1);
  ASSERT_EQ(size3 - size1, 2);
}

TEST(test_layer_factory, create) {
  using namespace kuiper_infer;
  LayerRegisterer::RegisterCreator("test3", TestCreateLayer);
  std::shared_ptr<RuntimeOperator> op = std::make_shared<RuntimeOperator>();
  op->type = "test3";
  std::shared_ptr<Layer> layer = LayerRegisterer::CreateLayer(op);
  ASSERT_EQ(layer->layer_name(),"test3");
}