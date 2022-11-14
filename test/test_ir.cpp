//
// Created by fss on 22-11-13.
//

#include <gtest/gtest.h>
#include "parser/runtime_ir.hpp"

TEST(test_ir, Load) {
  using namespace kuiper_infer;
  RuntimeGraph graph("./tmp/netjt.pnnx.param", "./tmp/netjt.pnnx.bin");
  graph.Init();
  int c = 3;
}