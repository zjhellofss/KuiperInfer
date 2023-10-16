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

// Created by fss on 23-2-27.
#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {
void RuntimeOperatorUtils<float>::InitOperatorInput(
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  if (operators.empty()) {
    LOG(ERROR) << "Operators for init input shapes is empty!";
    return;
  }

  for (const auto& op : operators) {
    if (op->input_operands.empty()) {
      continue;
    } else {
      const std::map<std::string, std::shared_ptr<RuntimeOperand>>& input_operands_map =
          op->input_operands;
      // 初始化operator的输入空间
      for (const auto& [_, input_operand] : input_operands_map) {
        if (!input_operand) {
          continue;
        }
        const auto& type = input_operand->type;
        auto& input_datas = input_operand->datas;  // 需要初始化的输入空间
        CHECK(type == RuntimeDataType::kTypeFloat32) << "The graph only support float32 yet!";
        const auto& input_operand_shape = input_operand->shapes;

        CHECK(!input_operand_shape.empty());
        const int32_t batch = input_operand_shape.at(0);
        CHECK(batch > 0) << "Dynamic batch size is not supported!";
        CHECK(input_operand_shape.size() == 2 || input_operand_shape.size() == 4 ||
              input_operand_shape.size() == 3)
            << "Unsupported tensor shape sizes: " << input_operand_shape.size();

        if (!input_datas.empty()) {
          CHECK_EQ(input_datas.size(), batch);
        } else {
          input_datas.resize(batch);
        }
      }
    }
  }
}

void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty() && pnnx_operators.size() == operators.size());
  CHECK(pnnx_operators.size() == operators.size());
  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    // 得到pnnx原有的输出空间
    const std::vector<pnnx::Operand*> operands = pnnx_operators.at(i)->outputs;
    const uint32_t operands_size = operands.size();
    if (!operands_size) {
      continue;
    } else if (operands_size > 1) {
      LOG(FATAL) << "Only support one node one output yet!";
    } else {
      // 一个节点仅支持一个输出，实际上在pnnx中一个节点拥有两个不同输出的情况也是不存在的
      pnnx::Operand* operand = operands.front();
      const auto& runtime_op = operators.at(i);

      CHECK(operand != nullptr && !operand->shape.empty()) << "Operand output is null or empty!";
      std::vector<int32_t> operand_shapes;
      for (int32_t dim : operand->shape) {
        CHECK_GT(dim, 0);
        operand_shapes.push_back(dim);
      }

      const int32_t batch = operand_shapes.front();
      const auto& output_tensors = runtime_op->output_operands;
      CHECK(batch > 0) << "Dynamic batch size is not supported!";
      CHECK(operand_shapes.size() == 2 || operand_shapes.size() == 4 || operand_shapes.size() == 3)
          << "Unsupported shape sizes: " << operand_shapes.size();
      CHECK_EQ(operand->type, 1) << "The type of pnnx operand is not float32";

      if (!output_tensors) {
        std::vector<std::shared_ptr<Tensor<float>>> output_operand_datas;
        if (runtime_op) {
          for (uint32_t j = 0; j < batch; ++j) {
            sftensor tensor;
            switch (operand_shapes.size()) {
              case 4: {
                tensor = TensorCreate<float>(operand_shapes.at(1), operand_shapes.at(2),
                                             operand_shapes.at(3));
                break;
              }
              case 3: {
                tensor = TensorCreate<float>(operand_shapes.at(1), operand_shapes.at(2));
                break;
              }
              case 2: {
                tensor = TensorCreate<float>(operand_shapes.at(1));
                break;
              }
              default: {
                LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
                break;
              }
            }
            output_operand_datas.push_back(tensor);
          }
          runtime_op->output_operands =
              std::make_shared<RuntimeOperand>(operand->name + "_output", operand_shapes,
                                               output_operand_datas, RuntimeDataType::kTypeFloat32);
        }
      } else {
        CHECK(batch == output_tensors->datas.size());
        CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
        CHECK(output_tensors->shapes == operand_shapes);
        // 逐批次检查输出空间的形状是否合理，如果不合理则进行reshape
        for (uint32_t b = 0; b < batch; ++b) {
          sftensor output_tensor = output_tensors->datas.at(b);
          const std::vector<uint32_t>& tensor_shapes = output_tensor->shapes();
          switch (operand_shapes.size()) {
            case 4: {
              if (tensor_shapes.at(0) != operand_shapes.at(1) ||
                  tensor_shapes.at(1) != operand_shapes.at(2) ||
                  tensor_shapes.at(2) != operand_shapes.at(3)) {
                output_tensor->Reshape({(uint32_t)operand_shapes.at(1),
                                        (uint32_t)operand_shapes.at(2),
                                        (uint32_t)operand_shapes.at(3)});
              }
              break;
            }
            case 3: {
              if (tensor_shapes.at(0) != 1 || tensor_shapes.at(1) != operand_shapes.at(1) ||
                  tensor_shapes.at(2) != operand_shapes.at(2)) {
                output_tensor->Reshape(
                    {(uint32_t)operand_shapes.at(1), (uint32_t)operand_shapes.at(2)});
              }
              break;
            }
            case 2: {
              if (tensor_shapes.at(0) != 1 || tensor_shapes.at(1) != operand_shapes.at(1) ||
                  tensor_shapes.at(2) != 1) {
                output_tensor->Reshape({(uint32_t)operand_shapes.at(1)});
              }
              break;
            }
            default: {
              LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
              break;
            }
          }
        }
      }
    }
  }
}

}  // namespace kuiper_infer
