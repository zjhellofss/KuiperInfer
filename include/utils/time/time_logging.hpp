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
#include <string>
#include <utility>
namespace kuiper_infer {
namespace utils {
using Time = std::chrono::steady_clock;

/**
 * @brief Timing state for a layer type
 *
 * Contains timing information for a layer type, including the
 * layer name, type, and duration.
 */
struct LayerTimeState {
  /**
   * @brief Construct new LayerTimeState
   *
   * @param duration_time Duration in ns
   * @param layer_name Name of layer
   * @param layer_type Type of layer
   */
  explicit LayerTimeState(long duration_time, std::string layer_name,
                          std::string layer_type);

  /// Duration in ns
  long duration_time_;

  /// Mutex for thread safety
  std::mutex time_mutex_;

  /// Name of the layer
  std::string layer_name_;

  /// Type of the layer
  std::string layer_type_;
};

/**
 * @brief Map of layer times by type
 *
 * Maps layer type string to LayerTimeState pointer.
 */
using LayerTimeStatesCollector =
    std::map<std::string, std::shared_ptr<LayerTimeState>>;

/**
 * @brief Pointer to layer time collector map
 */
using PtrLayerTimeStatesCollector = std::shared_ptr<LayerTimeStatesCollector>;

/**
 * @brief Singleton for layer timing stats
 *
 * Manages a singleton instance of a collector that stores
 * timing information per layer type. Thread safe.
 */
class LayerTimeStatesSingleton {
 public:
  LayerTimeStatesSingleton() = default;

  /**
   * @brief Initializes the singleton instance
   *
   * Creates the singleton collector instance.
   */
  static void LayerTimeStatesCollectorInit();

  /**
   * @brief Gets the singleton instance
   *
   * @return The singleton collector instance
   */
  static PtrLayerTimeStatesCollector SingletonInstance();

 private:
  static std::mutex mutex_;
  static PtrLayerTimeStatesCollector time_states_collector_;
};

/**
 * @brief Logs timing for a layer
 *
 * Logs the start and end time for executing a layer to compute
 * the layer duration. Collects timing per layer type.
 */
class LayerTimeLogging {
 public:
  /**
   * @brief Construct and start time log
   *
   * @param layer_name Layer name
   * @param layer_type Layer type
   */
  explicit LayerTimeLogging(std::string layer_name, std::string layer_type);

  /**
   * @brief Stop time log and record duration
   *
   * Records execution duration to singleton collector.
   */
  ~LayerTimeLogging();

  /**
   * @brief Prints summary of layer times
   *
   * Prints collected per layer type execution times.
   */
  static void SummaryLogging();

 private:
  std::string layer_name_;
  std::string layer_type_;
  std::chrono::steady_clock::time_point start_time_;
};
}  // namespace utils
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_UTILS_TIME_LOGGING_HPP_
