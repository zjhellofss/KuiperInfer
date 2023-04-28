//
// Created by fss on 23-4-27.
//

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
