//
// Created by fss on 22-11-12.
//
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include <glog/logging.h>

TEST(test_json, dump) {
  using namespace nlohmann;
  json json_data = {{"one", 1}, {"two", 2}};
  const auto &kvs = json_data.get<json::object_t>();
  ASSERT_EQ(kvs.size(), 2);
  ASSERT_EQ(kvs.find("one")->second, 1);
  ASSERT_EQ(kvs.find("two")->second, 2);
}