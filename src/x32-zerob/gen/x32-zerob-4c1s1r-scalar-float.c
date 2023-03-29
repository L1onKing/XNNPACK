// Auto-generated file. Do not edit!
//   Template: src/x32-packb/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/packb.h>
#include <xnnpack/unaligned.h>

void xnn_x32_zerob_gemm_ukernel_4c1s1r__scalar_float(
  size_t groups,
  size_t channels,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const union xnn_x32_packb_params* params)
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  float* out = (float*) packed_weights;
  const float zero = 0;
  do {
    // channel tile loop multiple of 4
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      unaligned_indexed_store_f32(out, 0, zero);
      unaligned_indexed_store_f32(out, 1, zero);
      unaligned_indexed_store_f32(out, 2, zero);
      unaligned_indexed_store_f32(out, 3, zero);

      out = (float*) ((uintptr_t) out + channel_tile_stride);
    }

    // channel subtile loop multiple of 1
    for (; c >= 1; c -= 1) {
      unaligned_indexed_store_f32(out, 0, zero);

      out = (float*) ((uintptr_t) out + channel_subtile_stride);
    }

  } while (--groups != 0);
}
