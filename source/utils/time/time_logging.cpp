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

// Created by fss on 23-4-27.

#include "utils/time/time_logging.hpp"
#include <utility>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
namespace utils {
PtrLayerTimeStatesCollector LayerTimeStatesSingleton::SingletonInstance() {
  std::lock_guard<std::mutex> lock_(mutex_);
  if (time_states_collector_ == nullptr) {
    time_states_collector_ = std::make_shared<LayerTimeStatesCollector>();
  }
  return time_states_collector_;
}

void LayerTimeStatesSingleton::LayerTimeStatesCollectorInit() {
  if (time_states_collector_ != nullptr) {
    std::lock_guard<std::mutex> lock_(mutex_);
    time_states_collector_.reset();
    time_states_collector_ = nullptr;
  }
  time_states_collector_ = LayerTimeStatesSingleton::SingletonInstance();
}

std::mutex LayerTimeStatesSingleton::mutex_;

PtrLayerTimeStatesCollector LayerTimeStatesSingleton::time_states_collector_;

LayerTimeLogging::LayerTimeLogging(std::string layer_name, std::string layer_type)
    : layer_name_(std::move(layer_name)),
      layer_type_(std::move(layer_type)),
      start_time_(Time::now()) {
  auto layer_time_states = LayerTimeStatesSingleton::SingletonInstance();
  layer_time_states->insert(
      {layer_name_, std::make_shared<LayerTimeState>(0l, layer_name_, layer_type_)});
}

LayerTimeLogging::~LayerTimeLogging() {
  auto layer_time_states = LayerTimeStatesSingleton::SingletonInstance();
  const auto layer_state_iter = layer_time_states->find(layer_name_);
  if (layer_state_iter != layer_time_states->end()) {
    auto& layer_state = layer_state_iter->second;
    CHECK(layer_state != nullptr);

    std::lock_guard<std::mutex> lock_guard(layer_state->time_mutex_);
    const auto end_time = Time::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    layer_state->duration_time_ += duration;
  } else {
    LOG(ERROR) << "Can not find the layer: " << layer_name_ << " in the time logging.";
  }
}

void LayerTimeLogging::SummaryLogging() {
  auto layer_time_states = LayerTimeStatesSingleton::SingletonInstance();
  CHECK(layer_time_states != nullptr);
  LayerTimeStatesCollector layer_time_states_collector = *layer_time_states.get();

  long total_time_costs = 0;
  for (const auto& [layer_name, layer_time_state] : layer_time_states_collector) {
    CHECK(layer_time_state != nullptr);

    std::lock_guard<std::mutex> lock(layer_time_state->time_mutex_);
    const auto time_cost = layer_time_state->duration_time_;
    total_time_costs += time_cost;
    if (layer_time_state->duration_time_ != 0) {
      LOG(INFO) << "Layer name: " << layer_name << "\t"
                << "layer type: " << layer_time_state->layer_type_ << "\t"
                << "time cost: " << time_cost << "ms";
    }
  }
  LOG(INFO) << "Total time: " << total_time_costs << "ms";
}

}  // namespace utils
}  // namespace kuiper_infer