// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 23-1-29.
#include <gtest/gtest.h>
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_runtime, runtime_graph_input_init1) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto& runtime_operator = std::make_shared<RuntimeOperator>();
    const auto& runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {3, 32, 32};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto& runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {3, 64, 64};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 3);
    ASSERT_EQ(size1, size2);
  }

  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 3);
    ASSERT_EQ(size1, size2);
  }
}

TEST(test_runtime, runtime_graph_input_init2) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto& runtime_operator = std::make_shared<RuntimeOperator>();
    const auto& runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {4, 3, 32, 32};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto& runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {4, 3, 64, 64};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 4);
    ASSERT_EQ(size1, size2);
  }

  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 4);
    ASSERT_EQ(size1, size2);
  }
}

TEST(test_runtime, runtime_graph_input_init3) {
  using namespace kuiper_infer;
  std::vector<std::shared_ptr<RuntimeOperator>> operators;
  uint32_t op_size = 3;
  for (uint32_t i = 0; i < op_size; ++i) {
    const auto& runtime_operator = std::make_shared<RuntimeOperator>();
    const auto& runtime_operand1 = std::make_shared<RuntimeOperand>();
    runtime_operand1->shapes = {5, 32};
    runtime_operand1->type = RuntimeDataType::kTypeFloat32;

    const auto& runtime_operand2 = std::make_shared<RuntimeOperand>();
    runtime_operand2->shapes = {5, 64};
    runtime_operand2->type = RuntimeDataType::kTypeFloat32;

    runtime_operator->input_operands.insert({std::string("size1"), runtime_operand1});
    runtime_operator->input_operands.insert({std::string("size2"), runtime_operand2});

    operators.push_back(runtime_operator);
  }
  ASSERT_EQ(operators.size(), 3);
  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 5);
    ASSERT_EQ(size1, size2);
  }

  RuntimeOperatorUtils<float>::InitOperatorInput(operators);
  for (uint32_t i = 0; i < operators.size(); ++i) {
    const auto& op = operators.at(i);
    ASSERT_EQ(op->input_operands["size1"]->datas.empty(), false);
    const uint32_t size1 = op->input_operands["size1"]->datas.size();
    const uint32_t size2 = op->input_operands["size2"]->datas.size();
    ASSERT_EQ(size1, 5);
    ASSERT_EQ(size1, size2);
  }
}

TEST(test_runtime, runtime_graph_output_init1) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 3, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 3);
      ASSERT_EQ(raw_shapes.at(0), 3);
      ASSERT_EQ(raw_shapes.at(1), 32);
      ASSERT_EQ(raw_shapes.at(2), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
      output_data->data().resize(32, 16, 6);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 3);
      ASSERT_EQ(raw_shapes.at(0), 3);
      ASSERT_EQ(raw_shapes.at(1), 32);
      ASSERT_EQ(raw_shapes.at(2), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init2) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 64};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 8);

    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 1);
      ASSERT_EQ(raw_shapes.at(0), 64);

      ASSERT_EQ(output_data->rows(), 1);
      ASSERT_EQ(output_data->cols(), 64);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(32, 1, 2);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 8);

    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 1);
      ASSERT_EQ(raw_shapes.at(0), 64);
      ASSERT_EQ(output_data->rows(), 1);
      ASSERT_EQ(output_data->cols(), 64);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init3) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 2);
      ASSERT_EQ(raw_shapes.at(0), 32);
      ASSERT_EQ(raw_shapes.at(1), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(32, 1, 1 * 32);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 2);
      ASSERT_EQ(raw_shapes.at(0), 32);
      ASSERT_EQ(raw_shapes.at(1), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init6) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{2, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 2);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 1);
      ASSERT_EQ(raw_shapes.at(0), 32);

      ASSERT_EQ(output_data->rows(), 1);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(1, 16, 2);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 2);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 1);
      ASSERT_EQ(raw_shapes.at(0), 32);

      ASSERT_EQ(output_data->rows(), 1);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init4) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 2);
      ASSERT_EQ(raw_shapes.at(0), 32);
      ASSERT_EQ(raw_shapes.at(1), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(32 * 32, 1, 1);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 2);
      ASSERT_EQ(raw_shapes.at(0), 32);
      ASSERT_EQ(raw_shapes.at(1), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init7) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 3, 31, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 3);
      ASSERT_EQ(raw_shapes.at(0), 3);
      ASSERT_EQ(raw_shapes.at(1), 31);
      ASSERT_EQ(raw_shapes.at(2), 32);

      ASSERT_EQ(output_data->rows(), 31);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
      output_data->data().resize(31, 16, 6);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 3);
      ASSERT_EQ(raw_shapes.at(0), 3);
      ASSERT_EQ(raw_shapes.at(1), 31);
      ASSERT_EQ(raw_shapes.at(2), 32);

      ASSERT_EQ(output_data->rows(), 31);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init8) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 363};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 1);
      ASSERT_EQ(raw_shapes.at(0), 363);

      ASSERT_EQ(output_data->rows(), 1);
      ASSERT_EQ(output_data->cols(), 363);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(3, 121, 1);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 2);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 1);
      ASSERT_EQ(raw_shapes.at(0), 363);

      ASSERT_EQ(output_data->rows(), 1);
      ASSERT_EQ(output_data->cols(), 363);
      ASSERT_EQ(output_data->channels(), 1);
    }
  }
}

TEST(test_runtime, runtime_graph_output_init9) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator*> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator* pnnx_op = new pnnx::Operator;
    pnnx::Operand* pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{7, 31, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 7);
    for (auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 2);
      ASSERT_EQ(raw_shapes.at(0), 31);
      ASSERT_EQ(raw_shapes.at(1), 32);

      ASSERT_EQ(output_data->rows(), 31);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 1);
      output_data->data().resize(16 * 31, 1, 2);
    }
  }

  RuntimeOperatorUtils<float>::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto& run_op : run_ops) {
    const auto& output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 3);
    ASSERT_EQ(output_datas->datas.size(), 7);
    for (const auto& output_data : output_datas->datas) {
      const auto& raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 2);
      ASSERT_EQ(raw_shapes.at(0), 31);
      ASSERT_EQ(raw_shapes.at(1), 32);

      ASSERT_EQ(output_data->rows(), 31);
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

TEST(test_runtime, graph_build1) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/add/resnet_add.pnnx.param", "tmp/add/resnet_add.pnnx.bin");
  ASSERT_EQ(int(graph.graph_state()), -2);  // need_init
  graph.Build();
  ASSERT_EQ(int(graph.graph_state()), 0);
}

TEST(test_runtime, op_is_input) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/add/resnet_add.pnnx.param", "tmp/add/resnet_add.pnnx.bin");
  ASSERT_EQ(int(graph.graph_state()), -2);  // need_init
  graph.Build();
  ASSERT_EQ(int(graph.graph_state()), 0);
  ASSERT_EQ(graph.is_input_op("pnnx_input_0"), true);
  ASSERT_EQ(graph.is_input_op("random_str"), false);
}

TEST(test_runtime, op_is_output) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/add/resnet_add.pnnx.param", "tmp/add/resnet_add.pnnx.bin");
  ASSERT_EQ(int(graph.graph_state()), -2);  // need_init
  graph.Build();
  ASSERT_EQ(int(graph.graph_state()), 0);
  ASSERT_EQ(graph.is_output_op("pnnx_output_0"), true);
  ASSERT_EQ(graph.is_output_op("random_str"), false);
}
