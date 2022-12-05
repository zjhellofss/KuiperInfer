//
// Created by fss on 22-12-5.
//

#include "data/fast_copy.hpp"
#include <cstring>
#include <glog/logging.h>
#include <x86intrin.h>

namespace kuiper_infer {

void *FastMemoryCopy(float *out, const float *in, size_t n) {
  CHECK(out != nullptr && in != nullptr);
  n >>= 4;
  for (__m512 *ov = (__m512 *) out, *iv = (__m512 *) in; n--;) {
    _mm512_storeu_ps((float *) (ov++), _mm512_loadu_ps((float *) (iv++)));
  }
  return out;
}
}