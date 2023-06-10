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

#ifndef KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
#define KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include "layer/abstract/layer_factory.hpp"
namespace kuiper_infer {
namespace utils {
using Time = std::chrono::steady_clock;

struct LayerTimeState {
  explicit LayerTimeState(long duration_time, const std::string& layer_type)
      : duration_time_(duration_time), layer_type_(layer_type) {}

  long duration_time_;
  std::mutex time_mutex_;
  std::string layer_type_;
};

using PtrLayerTimeStates =
    std::shared_ptr<std::map<std::string, std::shared_ptr<LayerTimeState>>>;

class LayerTimeStatesSingleton {
 public:
  LayerTimeStatesSingleton() = default;

  static PtrLayerTimeStates SingletonInstance();

  static void LayerTimeStatesInit();

 private:
  static std::mutex mutex_;
  static PtrLayerTimeStates layer_time_states_;
};



class LayerTimeLogging {
 public:
  explicit LayerTimeLogging(const std::string& layer_type);

  ~LayerTimeLogging();

  void SummaryLogging();

 private:
  std::string layer_type_;
  std::chrono::steady_clock::time_point start_time_;
  PtrLayerTimeStates layer_time_states_;
};
}  // namespace utils
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
