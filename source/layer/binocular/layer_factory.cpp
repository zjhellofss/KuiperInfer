//
// Created by fss on 22-11-17.
//

#include "layer_factory.hpp"
#include "parser/runtime_ir.hpp"

namespace kuiper_infer {
void LayerRegisterer::RegisterCreator(const std::string &layer_type, const Creator &creator) {
  CHECK(creator != nullptr);
  CreateRegistry &registry = Registry();
  CHECK_EQ(registry.count(layer_type), 0) << "Layer type: " << layer_type << " has already registered!";
  registry.insert({layer_type, creator});
}

LayerRegisterer::CreateRegistry &LayerRegisterer::Registry() {
  static CreateRegistry *kRegistry = new CreateRegistry;
  CHECK(kRegistry != nullptr);
  return *kRegistry;
}

std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<RuntimeOperator> &op,
                                                    std::shared_ptr<Layer> &layer) {
  CreateRegistry &registry = Registry();
  const std::string &layer_type = op->type;
  LOG_IF(FATAL, registry.count(layer_type) <= 0) << "Can not find the layer type: " << layer_type;
  const auto &creator = registry.find(layer_type)->second;
  const auto &status = creator(op, layer);
  LOG_IF(FATAL, status != ParseParameterAttrStatus::kParameterParseSuccess) << "Create the layer: " << layer_type
                                                                            << " failed, error code: " << int(status);
  return layer;
}
}
