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

void xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int(
  size_t groups,
  size_t channels,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const union xnn_x32_packb_params* params)
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  uint32_t* out = (uint32_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  do {
    // channel tile loop multiple of 4
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      unaligned_indexed_store_u32(out, 0, b[0]);
      unaligned_indexed_store_u32(out, 1, b[1]);
      unaligned_indexed_store_u32(out, 2, b[2]);
      unaligned_indexed_store_u32(out, 3, b[3]);
      b += 4;

      out = (uint32_t*) ((uintptr_t) out + channel_tile_stride);
    }


    // channels remainder (1..3)
    if XNN_UNLIKELY(c != 0) {
      if (c & 2) {
        unaligned_indexed_store_u32(out, 0, b[0]);
        unaligned_indexed_store_u32(out, 1, b[1]);
        b += 2;
        out += 2;
      }
      if (c & 1) {
        unaligned_indexed_store_u32(out, 0, b[0]);
        b += 1;
        out += 1;
      }

      out = (uint32_t*) ((uintptr_t) out + channel_subtile_stride - c * sizeof(uint32_t));
    }
  } while (--groups != 0);
}
