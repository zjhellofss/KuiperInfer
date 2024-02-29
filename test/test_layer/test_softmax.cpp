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

// Created by fss on 22-12-19.

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/softmax.hpp"
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_layer, forward_softmax_dim1) {
  // softmax on dim = 1
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/softmax/softmax_dim1.pnnx.param", "tmp/softmax/softmax_dim1.pnnx.bin");

  graph.Build();
  const uint32_t size = 24;
  std::vector<sftensor> inputs;
  std::vector<float> values;
  for (uint32_t i = 0; i < size; ++i) {
    values.push_back(float(i));
  }

  const uint32_t batch_size = 4;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(2, 3, 4);
    input->Fill(values);
    inputs.push_back(input);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");

  arma::fmat real = CSVDataLoader::LoadData<float>("tmp/softmax/softmax_dim1.csv");
  for (const auto& output : outputs) {
    output->Reshape({24}, true);
    for (int i = 0; i < 24; ++i) {
      float a = output->index(i);
      float b = real.at(i);
      ASSERT_LE(std::abs(a - b), 1e-5f) << "a: " << a << " b: " << b;
    }
  }
}

TEST(test_layer, forward_softmax_dim1_minus2) {
  // softmax on dim = 1
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/softmax/softmax_dim1_-2.pnnx.param",
                     "tmp/softmax/softmax_dim1_-2.pnnx.bin");

  graph.Build();
  const uint32_t size = 24;
  std::vector<sftensor> inputs;
  std::vector<float> values;
  for (uint32_t i = 0; i < size; ++i) {
    values.push_back(float(i));
  }

  const uint32_t batch_size = 4;
  for (uint32_t i = 0; i < batch_size; ++i) {
    sftensor input = std::make_shared<ftensor>(2, 3, 4);
    input->Fill(values);
    inputs.push_back(input);
  }
  graph.set_inputs("pnnx_input_0", inputs);
  graph.Forward(false);
  std::vector<std::shared_ptr<Tensor<float>>> outputs = graph.get_outputs("pnnx_output_0");

  arma::fmat real = CSVDataLoader::LoadData<float>("tmp/softmax/softmax_dim1.csv");
  for (const auto& output : outputs) {
    output->Reshape({24}, true);
    for (int i = 0; i < 24; ++i) {
      float a = output->index(i);
      float b = real.at(i);
      ASSERT_LE(std::abs(a - b), 1e-5f) << "a: " << a << " b: " << b << " i:" << i;
    }
  }
}

TEST(test_layer, forward_softmax_dim0) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(0);
  for (int k = 0; k < 2; ++k) {
    uint32_t size = 24;
    std::vector<float> values;
    for (uint32_t i = 0; i < size; ++i) {
      values.push_back(float(i));
    }

    const uint32_t batch_size = 4;
    std::vector<sftensor> inputs;

    for (uint32_t i = 0; i < batch_size; ++i) {
      sftensor input = std::make_shared<ftensor>(2, 3, 4);
      input->Fill(values);
      inputs.push_back(input);
    }
    std::vector<sftensor> outputs(batch_size);
    softmax_layer.Forward(inputs, outputs);

    arma::fmat real = CSVDataLoader::LoadData<float>("tmp/softmax/softmax_dim0.csv");
    for (const auto& output : outputs) {
      output->Reshape({24}, true);
      for (int i = 0; i < 24; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}

TEST(test_layer, forward_softmax_dim2) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(2);
  for (int k = 0; k < 2; ++k) {
    uint32_t size = 24;
    std::vector<float> values;
    for (uint32_t i = 0; i < size; ++i) {
      values.push_back(float(i));
    }

    const uint32_t batch_size = 4;
    std::vector<sftensor> inputs;

    for (uint32_t i = 0; i < batch_size; ++i) {
      sftensor input = std::make_shared<ftensor>(2, 3, 4);
      input->Fill(values);
      inputs.push_back(input);
    }
    std::vector<sftensor> outputs(batch_size);
    softmax_layer.Forward(inputs, outputs);

    arma::fmat real = CSVDataLoader::LoadData<float>("tmp/softmax/softmax_dim2.csv");
    for (const auto& output : outputs) {
      output->Reshape({24}, true);
      for (int i = 0; i < 24; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}

void softmax1(float* x, int size) {
  float* y = new float[size];
  for (int i = 0; i < size; ++i) {
    y[i] = x[i];
  }

  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

TEST(test_layer, forward_softmax_dim01) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(0);
  uint32_t size = 23;
  std::vector<float> values;
  for (uint32_t i = 0; i < size; ++i) {
    values.push_back(float(i));
  }

  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;

  sftensor input = std::make_shared<ftensor>(23);
  input->Fill(values);

  std::vector<sftensor> outputs(1);
  inputs.push_back(input);
  softmax_layer.Forward(inputs, outputs);
  softmax1(values.data(), values.size());
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input->index(i), float(i));
    ASSERT_NEAR(values.at(i), outputs.front()->index(i), 1e-3f);
  }
}


TEST(test_layer, forward_softmax_dim011) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(0);
  uint32_t size = 97;
  std::vector<float> values;
  for (uint32_t i = 0; i < size; ++i) {
    values.push_back(float(i));
  }

  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;

  sftensor input = std::make_shared<ftensor>(97);
  input->Fill(values);

  std::vector<sftensor> outputs(1);
  inputs.push_back(input);
  softmax_layer.Forward(inputs, outputs);
  softmax1(values.data(), values.size());
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input->index(i), float(i));
    ASSERT_NEAR(values.at(i), outputs.front()->index(i), 1e-3f);
  }
}

TEST(test_layer, forward_softmax_dim02) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(0);
  uint32_t size = 3;
  std::vector<float> values;
  for (uint32_t i = 0; i < size; ++i) {
    values.push_back(float(i));
  }

  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;

  sftensor input = std::make_shared<ftensor>(3);
  input->Fill(values);

  std::vector<sftensor> outputs(1);
  inputs.push_back(input);
  softmax_layer.Forward(inputs, outputs);
  softmax1(values.data(), values.size());
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input->index(i), float(i));
    ASSERT_NEAR(values.at(i), outputs.front()->index(i), 1e-3f);
  }
}

TEST(test_layer, forward_softmax_dim023) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(0);
  uint32_t size = 1;
  std::vector<float> values;
  for (uint32_t i = 0; i < size; ++i) {
    values.push_back(float(i + 3.1415f));
  }

  const uint32_t batch_size = 1;
  std::vector<sftensor> inputs;

  sftensor input = std::make_shared<ftensor>(1);
  input->Fill(values);

  std::vector<sftensor> outputs(1);
  inputs.push_back(input);
  softmax_layer.Forward(inputs, outputs);
  softmax1(values.data(), values.size());
  for (int i = 0; i < size; ++i) {
    ASSERT_EQ(input->index(i), i + 3.1415f);
    ASSERT_NEAR(values.at(i), outputs.front()->index(i), 1e-3f);
  }
}

TEST(test_layer, forward_softmax_dim1_1) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(0);
  for (int k = 0; k < 2; ++k) {
    uint32_t size = 24;
    std::vector<float> values;
    for (uint32_t i = 0; i < size; ++i) {
      values.push_back(float(i));
    }

    const uint32_t batch_size = 4;
    std::vector<sftensor> inputs;

    for (uint32_t i = 0; i < batch_size; ++i) {
      sftensor input = std::make_shared<ftensor>(1, 1, 24);
      input->Fill(values);
      inputs.push_back(input);
    }
    std::vector<sftensor> outputs(batch_size);
    softmax_layer.Forward(inputs, outputs);
    arma::fmat real = CSVDataLoader::LoadData<float>("tmp/softmax/softmax_dim1_1.csv");
    for (const auto& output : outputs) {
      output->Reshape({24}, true);
      for (int i = 0; i < 24; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}

TEST(test_layer, forward_softmax_dim1_1_m) {
  using namespace kuiper_infer;
  SoftmaxLayer softmax_layer(-1);
  for (int k = 0; k < 2; ++k) {
    uint32_t size = 24;
    std::vector<float> values;
    for (uint32_t i = 0; i < size; ++i) {
      values.push_back(float(i));
    }

    const uint32_t batch_size = 4;
    std::vector<sftensor> inputs;

    for (uint32_t i = 0; i < batch_size; ++i) {
      sftensor input = std::make_shared<ftensor>(1, 1, 24);
      input->Fill(values);
      inputs.push_back(input);
    }
    std::vector<sftensor> outputs(batch_size);
    softmax_layer.Forward(inputs, outputs);
    arma::fmat real = CSVDataLoader::LoadData<float>("tmp/softmax/softmax_dim1_1.csv");
    for (const auto& output : outputs) {
      output->Reshape({24}, true);
      for (int i = 0; i < 24; ++i) {
        float a = output->index(i);
        float b = real.at(i);
        ASSERT_LE(std::abs(a - b), 1e-5f);
      }
    }
  }
}