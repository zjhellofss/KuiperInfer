//
// Created by fss on 23-1-29.
//
#include <gtest/gtest.h>
#include "runtime/runtime_ir.hpp"

TEST(test_runtime, runtime_graph_input_init1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto &runtime_operator = std::make_shared<RuntimeOperator>();
    const auto &runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {3, 32, 32};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto &runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {3, 64, 64};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeGraphShape::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const auto &shape1 = op->input_operands["size1"]->datas.front()->shapes();
    ASSERT_EQ(shape1.size(), 3);
    ASSERT_EQ(shape1.at(0), 1);
    ASSERT_EQ(shape1.at(1), 32);
    ASSERT_EQ(shape1.at(2), 32);

    const auto &shape2 = op->input_operands["size2"]->datas.front()->shapes();
    ASSERT_EQ(shape2.size(), 3);
    ASSERT_EQ(shape2.at(0), 1);
    ASSERT_EQ(shape2.at(1), 64);
    ASSERT_EQ(shape2.at(2), 64);
  }
  RuntimeGraphShape::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const auto &shape1 = op->input_operands["size1"]->datas.front()->shapes();
    ASSERT_EQ(shape1.size(), 3);
    ASSERT_EQ(shape1.at(0), 1);
    ASSERT_EQ(shape1.at(1), 32);
    ASSERT_EQ(shape1.at(2), 32);

    const auto &shape2 = op->input_operands["size2"]->datas.front()->shapes();
    ASSERT_EQ(shape2.size(), 3);
    ASSERT_EQ(shape2.at(0), 1);
    ASSERT_EQ(shape2.at(1), 64);
    ASSERT_EQ(shape2.at(2), 64);
  }
}

TEST(test_runtime, runtime_graph_input_init2) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto &runtime_operator = std::make_shared<RuntimeOperator>();
    const auto &runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {1, 3, 32, 32};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto &runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {1, 3, 64, 64};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeGraphShape::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const auto &shape1 = op->input_operands["size1"]->datas.front()->shapes();
    ASSERT_EQ(shape1.size(), 3);
    ASSERT_EQ(shape1.at(0), 3);
    ASSERT_EQ(shape1.at(1), 32);
    ASSERT_EQ(shape1.at(2), 32);

    const auto &shape2 = op->input_operands["size2"]->datas.front()->shapes();
    ASSERT_EQ(shape2.size(), 3);
    ASSERT_EQ(shape2.at(0), 3);
    ASSERT_EQ(shape2.at(1), 64);
    ASSERT_EQ(shape2.at(2), 64);
  }
  RuntimeGraphShape::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const auto &shape1 = op->input_operands["size1"]->datas.front()->shapes();
    ASSERT_EQ(shape1.size(), 3);
    ASSERT_EQ(shape1.at(0), 3);
    ASSERT_EQ(shape1.at(1), 32);
    ASSERT_EQ(shape1.at(2), 32);

    const auto &shape2 = op->input_operands["size2"]->datas.front()->shapes();
    ASSERT_EQ(shape2.size(), 3);
    ASSERT_EQ(shape2.at(0), 3);
    ASSERT_EQ(shape2.at(1), 64);
    ASSERT_EQ(shape2.at(2), 64);
  }
}

TEST(test_runtime, runtime_graph_input_init3) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto &runtime_operator = std::make_shared<RuntimeOperator>();
    const auto &runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {1, 99};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto &runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {1, 97};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeGraphShape::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const auto &shape1 = op->input_operands["size1"]->datas.front()->shapes();
    ASSERT_EQ(shape1.size(), 3);
    ASSERT_EQ(shape1.at(0), 1);
    ASSERT_EQ(shape1.at(1), 99);
    ASSERT_EQ(shape1.at(2), 1);

    const auto &shape2 = op->input_operands["size2"]->datas.front()->shapes();
    ASSERT_EQ(shape2.size(), 3);
    ASSERT_EQ(shape2.at(0), 1);
    ASSERT_EQ(shape2.at(1), 97);
    ASSERT_EQ(shape2.at(2), 1);
  }
  RuntimeGraphShape::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const auto &shape1 = op->input_operands["size1"]->datas.front()->shapes();
    ASSERT_EQ(shape1.size(), 3);
    ASSERT_EQ(shape1.at(0), 1);
    ASSERT_EQ(shape1.at(1), 99);
    ASSERT_EQ(shape1.at(2), 1);

    const auto &shape2 = op->input_operands["size2"]->datas.front()->shapes();
    ASSERT_EQ(shape2.size(), 3);
    ASSERT_EQ(shape2.at(0), 1);
    ASSERT_EQ(shape2.at(1), 97);
    ASSERT_EQ(shape2.at(2), 1);
  }
}