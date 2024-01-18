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

static sftensor CreateTensor(const std::vector<int32_t>& operand_shapes) {
  switch (operand_shapes.size()) {
    case 4:
      return TensorCreate<float>(operand_shapes[1], operand_shapes[2], operand_shapes[3]);
    case 3:
      return TensorCreate<float>(operand_shapes[1], operand_shapes[2]);
    case 2:
      return TensorCreate<float>(operand_shapes[1]);
    default:
      LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
      return nullptr;
  }
}

static void CheckAndReshapeTensor(sftensor& output_tensor,
                                  const std::vector<int32_t>& operand_shapes) {
  const std::vector<uint32_t>& tensor_shapes = output_tensor->shapes();
  switch (operand_shapes.size()) {
    case 4:
      if (tensor_shapes[0] != operand_shapes[1] || tensor_shapes[1] != operand_shapes[2] ||
          tensor_shapes[2] != operand_shapes[3]) {
        output_tensor->Reshape({(uint32_t)operand_shapes[1], (uint32_t)operand_shapes[2],
                                (uint32_t)operand_shapes[3]});
      }
      break;
    case 3:
      if (tensor_shapes[0] != 1 || tensor_shapes[1] != operand_shapes[1] ||
          tensor_shapes[2] != operand_shapes[2]) {
        output_tensor->Reshape({(uint32_t)operand_shapes[1], (uint32_t)operand_shapes[2]});
      }
      break;
    case 2:
      if (tensor_shapes[0] != 1 || tensor_shapes[1] != operand_shapes[1] || tensor_shapes[2] != 1) {
        output_tensor->Reshape({(uint32_t)operand_shapes[1]});
      }
      break;
    default:
      LOG(FATAL) << "Unknown output operand shape length: " << operand_shapes.size();
      break;
  }
}

void RuntimeOperatorUtils<float>::InitOperatorOutput(
    const std::vector<pnnx::Operator*>& pnnx_operators,
    const std::vector<std::shared_ptr<RuntimeOperator>>& operators) {
  CHECK(!pnnx_operators.empty() && !operators.empty() && pnnx_operators.size() == operators.size());
  CHECK(pnnx_operators.size() == operators.size());
  for (uint32_t i = 0; i < pnnx_operators.size(); ++i) {
    const std::vector<pnnx::Operand*> operands = pnnx_operators[i]->outputs;
    if (operands.empty()) continue;
    if (operands.size() > 1) {
      LOG(FATAL) << "Only support one node one output yet!";
    }

    pnnx::Operand* operand = operands[0];
    CHECK(operand != nullptr && !operand->shape.empty()) << "Operand output is null or empty!";
    std::vector<int32_t> operand_shapes;
    std::copy_if(operand->shape.begin(), operand->shape.end(), std::back_inserter(operand_shapes),
                 [](int32_t dim) { return dim > 0; });
    const auto& runtime_op = operators[i];
    const auto& output_tensors = runtime_op->output_operands;
    CHECK((operand_shapes.size() == 2 || operand_shapes.size() == 4 || operand_shapes.size() == 3))
        << "Unsupported shape sizes: " << operand_shapes.size();

    const int32_t batch = operand_shapes[0];
    CHECK_EQ(operand->type, 1) << "The type of pnnx operand is not float32";
    if (!output_tensors) {
      std::vector<sftensor> output_operand_datas;
      for (uint32_t j = 0; j < batch; ++j) {
        output_operand_datas.push_back(CreateTensor(operand_shapes));
      }
      runtime_op->output_operands =
          std::make_shared<RuntimeOperand>(operand->name + "_output", operand_shapes,
                                           output_operand_datas, RuntimeDataType::kTypeFloat32);
    } else {
      CHECK(batch == output_tensors->datas.size());
      CHECK(output_tensors->type == RuntimeDataType::kTypeFloat32);
      CHECK(output_tensors->shapes == operand_shapes);
      for (uint32_t b = 0; b < batch; ++b) {
        sftensor output_tensor = output_tensors->datas[b];
        CheckAndReshapeTensor(output_tensor, operand_shapes);
      }
    }
  }
}

}  // namespace kuiper_infer
