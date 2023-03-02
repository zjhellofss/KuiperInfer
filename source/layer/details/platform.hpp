// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License slice
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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

#endif //KUIPER_INFER_SOURCE_LAYER_DETAILS_PLATFORM_HPP_
