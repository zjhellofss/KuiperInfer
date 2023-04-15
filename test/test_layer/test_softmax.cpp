//
// Created by fss on 22-12-19.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include "../../source/layer/details/softmax.hpp"
#include "data/load_data.hpp"
#include "runtime/runtime_ir.hpp"

TEST(test_layer, forward_softmax_dim1) {
  // softmax on dim = 1
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/softmax/softmax_dim1.pnnx.param",
                     "tmp/softmax/softmax_dim1.pnnx.bin");

  graph.Build("pnnx_input_0", "pnnx_output_0");
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
  const auto& outputs = graph.Forward(inputs);

  arma::fmat real = CSVDataLoader::LoadData("tmp/softmax/softmax_dim1.csv");
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

  graph.Build("pnnx_input_0", "pnnx_output_0");
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
  const auto& outputs = graph.Forward(inputs);

  arma::fmat real = CSVDataLoader::LoadData("tmp/softmax/softmax_dim1.csv");
  for (const auto& output : outputs) {
    output->Reshape({24}, true);
    for (int i = 0; i < 24; ++i) {
      float a = output->index(i);
      float b = real.at(i);
      ASSERT_LE(std::abs(a - b), 1e-5f)
          << "a: " << a << " b: " << b << " i:" << i;
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

    arma::fmat real = CSVDataLoader::LoadData("tmp/softmax/softmax_dim0.csv");
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