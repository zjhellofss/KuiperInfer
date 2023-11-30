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

// Created by fss on 22-11-17.

#ifndef KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#define KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
#include <map>
#include <memory>
#include <string>
#include "layer.hpp"
#include "runtime/runtime_op.hpp"

namespace kuiper_infer {
class LayerRegisterer {
 private:
  typedef StatusCode (*Creator)(const std::shared_ptr<RuntimeOperator>& op,
                                std::shared_ptr<Layer<float>>& layer);

  typedef std::map<std::string, Creator> CreateRegistry;

 public:
  friend class LayerRegistererWrapper;
  friend class RegistryGarbageCollector;

  /**
   * @brief Registers a layer creator function
   *
   * Registers a layer creation function for the given layer type.
   * The creation function generates a layer instance from runtime operator.
   *
   * @param layer_type The name of the layer type
   * @param creator Function to create the layer
   */
  static void RegisterCreator(const std::string& layer_type, const Creator& creator);

  /**
   * @brief Creates a layer object
   *
   * Calls the registered creator function for the given runtime operator
   * to create a layer instance.
   *
   * @param op The runtime operator
   * @return A shared pointer to the created layer object
   */
  static std::shared_ptr<Layer<float>> CreateLayer(const std::shared_ptr<RuntimeOperator>& op);

  /**
   * @brief Gets the layer registry
   *
   * @return Pointer to the layer creator registry
   */
  static CreateRegistry* Registry();

  /**
   * @brief Gets registered layer types
   *
   * @return A vector of registered layer type names
   */
  static std::vector<std::string> layer_types();

 private:
  static CreateRegistry* registry_;
};

/**
 * @brief Layer registry wrapper
 *
 * Helper class to register a layer creator function.
 * Automatically calls LayerRegisterer::RegisterCreator.
 */
class LayerRegistererWrapper {
 public:
  explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator,
                                  const std::string& layer_type) {
    LayerRegisterer::RegisterCreator(layer_type, creator);
  }

  template <typename... Ts>
  explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator,
                                  const std::string& layer_type, const Ts&... other_layer_types)
      : LayerRegistererWrapper(creator, other_layer_types...) {
    LayerRegisterer::RegisterCreator(layer_type, creator);
  }

  explicit LayerRegistererWrapper(const LayerRegisterer::Creator& creator) {}
};

/**
 * @brief Garbage collector for layer registry
 *
 * Destructor that cleans up the LayerRegisterer registry
 * when program exits. Deletes registry pointer if allocated.
 */
class RegistryGarbageCollector {
 public:
  ~RegistryGarbageCollector() {
    if (LayerRegisterer::registry_ != nullptr) {
      delete LayerRegisterer::registry_;
      LayerRegisterer::registry_ = nullptr;
    }
  }
  friend class LayerRegisterer;

 private:
  RegistryGarbageCollector() = default;
  RegistryGarbageCollector(const RegistryGarbageCollector&) = default;
};

}  // namespace kuiper_infer

#endif  // KUIPER_INFER_SOURCE_LAYER_LAYER_FACTORY_HPP_
