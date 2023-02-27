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

    runtime_operator->input_operands.insert(
        {std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert(
        {std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size1"]->datas) {
      const auto &shape1 = input_datas->shapes();
      ASSERT_EQ(shape1.size(), 3);
      ASSERT_EQ(shape1.at(0), 1);
      ASSERT_EQ(shape1.at(1), 32);
      ASSERT_EQ(shape1.at(2), 32);
    }

    ASSERT_EQ(op->input_operands["size2"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size2"]->datas) {
      const auto &shape2 = input_datas->shapes();
      ASSERT_EQ(shape2.size(), 3);
      ASSERT_EQ(shape2.at(0), 1);
      ASSERT_EQ(shape2.at(1), 64);
      ASSERT_EQ(shape2.at(2), 64);
    }
  }
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size1"]->datas) {
      const auto &shape1 = input_datas->shapes();
      ASSERT_EQ(shape1.size(), 3);
      ASSERT_EQ(shape1.at(0), 1);
      ASSERT_EQ(shape1.at(1), 32);
      ASSERT_EQ(shape1.at(2), 32);
    }

    ASSERT_EQ(op->input_operands["size2"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size2"]->datas) {
      const auto &shape2 = input_datas->shapes();
      ASSERT_EQ(shape2.size(), 3);
      ASSERT_EQ(shape2.at(0), 1);
      ASSERT_EQ(shape2.at(1), 64);
      ASSERT_EQ(shape2.at(2), 64);
    }
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

    runtime_operator->input_operands.insert(
        {std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert(
        {std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size1"]->datas) {
      const auto &shape1 = input_datas->shapes();
      ASSERT_EQ(shape1.size(), 3);
      ASSERT_EQ(shape1.at(0), 3);
      ASSERT_EQ(shape1.at(1), 32);
      ASSERT_EQ(shape1.at(2), 32);
    }

    ASSERT_EQ(op->input_operands["size2"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size2"]->datas) {
      const auto &shape2 = input_datas->shapes();
      ASSERT_EQ(shape2.size(), 3);
      ASSERT_EQ(shape2.at(0), 3);
      ASSERT_EQ(shape2.at(1), 64);
      ASSERT_EQ(shape2.at(2), 64);
    }
  }
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size1"]->datas) {
      const auto &shape1 = input_datas->shapes();
      ASSERT_EQ(shape1.size(), 3);
      ASSERT_EQ(shape1.at(0), 3);
      ASSERT_EQ(shape1.at(1), 32);
      ASSERT_EQ(shape1.at(2), 32);
    }

    ASSERT_EQ(op->input_operands["size2"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size2"]->datas) {
      const auto &shape2 = input_datas->shapes();
      ASSERT_EQ(shape2.size(), 3);
      ASSERT_EQ(shape2.at(0), 3);
      ASSERT_EQ(shape2.at(1), 64);
      ASSERT_EQ(shape2.at(2), 64);
    }
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

    runtime_operator->input_operands.insert(
        {std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert(
        {std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size1"]->datas) {
      const auto &shape1 = input_datas->shapes();
      ASSERT_EQ(shape1.size(), 3);
      ASSERT_EQ(shape1.at(0), 1);
      ASSERT_EQ(shape1.at(1), 99);
      ASSERT_EQ(shape1.at(2), 1);
    }

    ASSERT_EQ(op->input_operands["size2"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size2"]->datas) {
      const auto &shape2 = input_datas->shapes();
      ASSERT_EQ(shape2.size(), 3);
      ASSERT_EQ(shape2.at(0), 1);
      ASSERT_EQ(shape2.at(1), 97);
      ASSERT_EQ(shape2.at(2), 1);
    }
  }
  RuntimeOperatorUtils::InitOperatorInputTensor(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto &op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size1"]->datas) {
      const auto &shape1 = input_datas->shapes();
      ASSERT_EQ(shape1.size(), 3);
      ASSERT_EQ(shape1.at(0), 1);
      ASSERT_EQ(shape1.at(1), 99);
      ASSERT_EQ(shape1.at(2), 1);
    }

    ASSERT_EQ(op->input_operands["size2"]->datas.empty(), false);
    for (const auto &input_datas : op->input_operands["size2"]->datas) {
      const auto &shape2 = input_datas->shapes();
      ASSERT_EQ(shape2.size(), 3);
      ASSERT_EQ(shape2.at(0), 1);
      ASSERT_EQ(shape2.at(1), 97);
      ASSERT_EQ(shape2.at(2), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init1) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator *> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_number = new pnnx::Operand;
    pnnx_number->shape = std::vector<int>{8, 3, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
      output_data->data().resize(32, 16, 6);
    }
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init2) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator *> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_number = new pnnx::Operand;
    pnnx_number->shape = std::vector<int>{8, 64};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 8);

    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 64);
      ASSERT_EQ(output_data->cols(), 1);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(32, 1, 2);
    }
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 8);

    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 64);
      ASSERT_EQ(output_data->cols(), 1);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init3) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator *> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_number = new pnnx::Operand;
    pnnx_number->shape = std::vector<int>{8, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(32, 1, 1 * 32);
    }
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init4) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator *> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_number = new pnnx::Operand;
    pnnx_number->shape = std::vector<int>{8, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(32 * 32, 1, 1);
    }
  }

  RuntimeOperatorUtils::InitOperatorOutputTensor(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto &output_data : output_datas->datas) {
      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, set_param_path) {
  using namespace kuiper_infer;
  RuntimeGraph graph("xx.param", "yy.bin");
  ASSERT_EQ(graph.param_path(), "xx.param");
  graph.set_param_path("yy.param");
  ASSERT_EQ(graph.param_path(), "yy.param");
}

TEST(test_runtime, set_bin_path) {
  using namespace kuiper_infer;
  RuntimeGraph graph("xx.param", "yy.bin");
  ASSERT_EQ(graph.bin_path(), "yy.bin");
  graph.set_bin_path("yy.bin");
  ASSERT_EQ(graph.bin_path(), "yy.bin");
}