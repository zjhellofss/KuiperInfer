//
// Created by yizhu on 2023/2/5.
//

#include <gtest/gtest.h>
#include "runtime/runtime_parameter.hpp"

TEST(test_runtime, runtime_param1) {
  using namespace kuiper_infer;
  RuntimeParameter* param = new RuntimeParameterInt;
  ASSERT_EQ(param->type, RuntimeParameterType::kParameterInt);
}

TEST(test_runtime, runtime_param2) {
  using namespace kuiper_infer;
  RuntimeParameter* param = new RuntimeParameterInt;
  ASSERT_EQ(param->type, RuntimeParameterType::kParameterInt);
  ASSERT_EQ(dynamic_cast<RuntimeParameterFloat*>(param), nullptr);
  ASSERT_NE(dynamic_cast<RuntimeParameterInt*>(param), nullptr);
}