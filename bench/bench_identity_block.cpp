//
// Created by fss on 22-11-19.
//
#include <benchmark/benchmark.h>
#include "parser/runtime_ir.hpp"
#include "tick.hpp"

static void BM_Identity(benchmark::State &state) {
  using namespace kuiper_infer;
  RuntimeGraph graph("/home/fss/code/kuiper_infer/tmp/resnet181912.pnnx.param",
                     "/home/fss/code/kuiper_infer/tmp/resnet181912.pnnx.bin");
  bool init = graph.Init();
  CHECK(init == true);
  graph.Build();
  std::vector<std::shared_ptr<Tensor>> inputs;
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>(3, 512, 512);

  input->Fill(1.);
  inputs.push_back(input);
  for (auto _ : state) {
    graph.Forward(inputs, false);
  }
}
BENCHMARK(BM_Identity)->Iterations(5);
BENCHMARK_MAIN();