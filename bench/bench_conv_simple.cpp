//
// Created by fss on 22-11-22.
//
#include <benchmark/benchmark.h>
#include "parser/runtime_ir.hpp"
#include "data/load_data.hpp"
static void BM_ConvSimple(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("tmp/im2cols/convlarge.pnnx.param", "tmp/im2cols/convlarge.pnnx.bin");
  graph.Build();
  std::vector<std::shared_ptr<Tensor>> inputs;
  std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>(3, 512, 512);
  input_tensor->at(0) = CSVDataLoader::LoadData("./tmp/im2cols/1_large.csv");
  input_tensor->at(1) = CSVDataLoader::LoadData("./tmp/im2cols/2_large.csv");
  input_tensor->at(2) = CSVDataLoader::LoadData("./tmp/im2cols/3_large.csv");
  inputs.push_back(input_tensor);

  for (auto _ : state) {
    const auto &output = graph.Forward(inputs, false);
  }
}

BENCHMARK(BM_ConvSimple);
BENCHMARK_MAIN();