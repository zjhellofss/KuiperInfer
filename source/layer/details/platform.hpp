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
    
// available.
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// License slice
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// the License.

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_PLATFORM_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_PLATFORM_HPP_
#define KUIPER_FORCE_INLINE 1

#if KUIPER_FORCE_INLINE
#ifdef _MSC_VER
#define KUIPER_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define KUIPER_FORCEINLINE inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define KUIPER_FORCEINLINE inline __attribute__((__always_inline__))
#else
#define KUIPER_FORCEINLINE inline
#endif
#else
#define KUIPER_FORCEINLINE inline
#endif
#else
#define KUIPER_FORCEINLINE inline
#endif

#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_PLATFORM_HPP_
