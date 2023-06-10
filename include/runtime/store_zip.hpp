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
    
//
//
// in compliance with the License. You may obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software distributed
// CONDITIONS OF ANY KIND, either express or implied. See the License for the

#ifndef PNNX_STOREZIP_H
#define PNNX_STOREZIP_H

#include <map>
#include <string>
#include <vector>

namespace pnnx {

class StoreZipReader
{
 public:
  StoreZipReader();
  ~StoreZipReader();

  int open(const std::string& path);

  size_t get_file_size(const std::string& name);

  int read_file(const std::string& name, char* data);

  int close();

 private:
  FILE* fp;

  struct StoreZipMeta
  {
    size_t offset;
    size_t size;
  };

  std::map<std::string, StoreZipMeta> filemetas;
};

class StoreZipWriter
{
 public:
  StoreZipWriter();
  ~StoreZipWriter();

  int open(const std::string& path);

  int write_file(const std::string& name, const char* data, size_t size);

  int close();

 private:
  FILE* fp;

  struct StoreZipMeta
  {
    std::string name;
    size_t lfh_offset;
    uint32_t crc32;
    uint32_t size;
  };

  std::vector<StoreZipMeta> filemetas;
};

} // namespace pnnx

#endif // PNNX_STOREZIP_H
