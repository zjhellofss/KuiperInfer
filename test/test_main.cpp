//
// Created by fss on 22-11-17.
//
#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging("Kuiper");
  FLAGS_log_dir = "./log/";
  FLAGS_alsologtostderr = true;

  LOG(INFO) << "Start test...\n";
  return RUN_ALL_TESTS();
}