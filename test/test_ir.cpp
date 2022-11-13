//
// Created by fss on 22-11-13.
//

#include <gtest/gtest.h>
#include "parser/ir.hpp"

TEST(test_ir, Load) {
  using namespace kuiper_infer;
  Graph graph;
  bool open_success1 = graph.Load("", "");
  ASSERT_EQ(open_success1, false);

  bool open_success2 = graph.Load("./tmp/netjt.pnnx.param", "./tmp/netjt.pnnx.bin");
  ASSERT_EQ(open_success2, true);

}