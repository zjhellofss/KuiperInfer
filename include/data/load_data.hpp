//
// Created by fss on 22-11-21.
//

#ifndef KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#define KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
#include <armadillo>
#include <string>
namespace kuiper_infer {

class CSVDataLoader {
 public:
  static arma::fmat LoadData(const std::string &file_path, char split_char = ',');
 private:
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream &file, char split_char);
};
}

#endif //KUIPER_COURSE_INCLUDE_DATA_LOAD_DATA_HPP_
