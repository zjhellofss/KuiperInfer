//
// Created by fss on 22-12-5.
//

#ifndef KUIPER_INFER_INCLUDE_DATA_FAST_COPY_HPP_
#define KUIPER_INFER_INCLUDE_DATA_FAST_COPY_HPP_
#include <cstdlib>
namespace kuiper_infer{
extern "C"{
void *FastMemoryCopy(float *out, const float *in, size_t n);
}
}
#endif //KUIPER_INFER_INCLUDE_DATA_FAST_COPY_HPP_
