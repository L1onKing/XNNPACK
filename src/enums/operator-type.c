// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/operator-type.yaml
//   Generator: tools/generate-enum.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/operator-type.h>


static const uint16_t offset[133] = {
  0, 8, 22, 36, 50, 64, 78, 92, 119, 147, 175, 203, 230, 257, 275, 293, 318, 344, 360, 376, 391, 406, 428, 451, 474,
  497, 520, 543, 566, 584, 607, 631, 649, 672, 696, 720, 744, 768, 792, 816, 840, 854, 869, 884, 910, 936, 962, 988,
  1020, 1052, 1078, 1105, 1132, 1149, 1166, 1200, 1214, 1228, 1242, 1258, 1274, 1300, 1326, 1352, 1378, 1412, 1446,
  1480, 1514, 1548, 1582, 1612, 1642, 1662, 1682, 1703, 1724, 1745, 1766, 1790, 1814, 1837, 1860, 1878, 1896, 1914,
  1932, 1951, 1970, 1989, 2008, 2025, 2042, 2058, 2074, 2102, 2130, 2158, 2186, 2213, 2240, 2258, 2276, 2294, 2312,
  2327, 2343, 2359, 2377, 2395, 2413, 2439, 2466, 2493, 2510, 2527, 2549, 2571, 2600, 2629, 2648, 2667, 2686, 2705,
  2720, 2735, 2750, 2765, 2784, 2804, 2824, 2845, 2866
};

static const char data[] =
  "Invalid\0"
  "Abs (NC, F16)\0"
  "Abs (NC, F32)\0"
  "Add (ND, F16)\0"
  "Add (ND, F32)\0"
  "Add (ND, QS8)\0"
  "Add (ND, QU8)\0"
  "ArgMax Pooling (NHWC, F32)\0"
  "Average Pooling (NHWC, F16)\0"
  "Average Pooling (NHWC, F32)\0"
  "Average Pooling (NHWC, QU8)\0"
  "Bankers Rounding (NC, F16)\0"
  "Bankers Rounding (NC, F32)\0"
  "Ceiling (NC, F16)\0"
  "Ceiling (NC, F32)\0"
  "Channel Shuffle (NC, X8)\0"
  "Channel Shuffle (NC, X32)\0"
  "Clamp (NC, F16)\0"
  "Clamp (NC, F32)\0"
  "Clamp (NC, S8)\0"
  "Clamp (NC, U8)\0"
  "Constant Pad (ND, X8)\0"
  "Constant Pad (ND, X16)\0"
  "Constant Pad (ND, X32)\0"
  "Convert (NC, F16, F32)\0"
  "Convert (NC, F32, F16)\0"
  "Convert (NC, F32, QS8)\0"
  "Convert (NC, F32, QU8)\0"
  "Convert (NC, QS8)\0"
  "Convert (NC, QS8, F32)\0"
  "Convert (NC, QS16, QS8)\0"
  "Convert (NC, QU8)\0"
  "Convert (NC, QU8, F32)\0"
  "Convolution (NCHW, F16)\0"
  "Convolution (NCHW, F32)\0"
  "Convolution (NHWC, F16)\0"
  "Convolution (NHWC, F32)\0"
  "Convolution (NHWC, QC8)\0"
  "Convolution (NHWC, QS8)\0"
  "Convolution (NHWC, QU8)\0"
  "Copy (NC, X8)\0"
  "Copy (NC, X16)\0"
  "Copy (NC, X32)\0"
  "Deconvolution (NHWC, F16)\0"
  "Deconvolution (NHWC, F32)\0"
  "Deconvolution (NHWC, QS8)\0"
  "Deconvolution (NHWC, QU8)\0"
  "Depth To Space (NCHW2NHWC, X16)\0"
  "Depth To Space (NCHW2NHWC, X32)\0"
  "Depth To Space (NHWC, X8)\0"
  "Depth To Space (NHWC, X16)\0"
  "Depth To Space (NHWC, X32)\0"
  "Divide (ND, F16)\0"
  "Divide (ND, F32)\0"
  "Dynamic Fully Connected (NC, F32)\0"
  "ELU (NC, F16)\0"
  "ELU (NC, F32)\0"
  "ELU (NC, QS8)\0"
  "Floor (NC, F16)\0"
  "Floor (NC, F32)\0"
  "Fully Connected (NC, F16)\0"
  "Fully Connected (NC, F32)\0"
  "Fully Connected (NC, QS8)\0"
  "Fully Connected (NC, QU8)\0"
  "Global Average Pooling (NCW, F16)\0"
  "Global Average Pooling (NCW, F32)\0"
  "Global Average Pooling (NWC, F16)\0"
  "Global Average Pooling (NWC, F32)\0"
  "Global Average Pooling (NWC, QS8)\0"
  "Global Average Pooling (NWC, QU8)\0"
  "Global Sum Pooling (NWC, F16)\0"
  "Global Sum Pooling (NWC, F32)\0"
  "HardSwish (NC, F16)\0"
  "HardSwish (NC, F32)\0"
  "Leaky ReLU (NC, F16)\0"
  "Leaky ReLU (NC, F32)\0"
  "Leaky ReLU (NC, QS8)\0"
  "Leaky ReLU (NC, QU8)\0"
  "Max Pooling (NHWC, F16)\0"
  "Max Pooling (NHWC, F32)\0"
  "Max Pooling (NHWC, S8)\0"
  "Max Pooling (NHWC, U8)\0"
  "Maximum (ND, F16)\0"
  "Maximum (ND, F32)\0"
  "Minimum (ND, F16)\0"
  "Minimum (ND, F32)\0"
  "Multiply (ND, F16)\0"
  "Multiply (ND, F32)\0"
  "Multiply (ND, QS8)\0"
  "Multiply (ND, QU8)\0"
  "Negate (NC, F16)\0"
  "Negate (NC, F32)\0"
  "PReLU (NC, F16)\0"
  "PReLU (NC, F32)\0"
  "Resize Bilinear (NCHW, F16)\0"
  "Resize Bilinear (NCHW, F32)\0"
  "Resize Bilinear (NHWC, F16)\0"
  "Resize Bilinear (NHWC, F32)\0"
  "Resize Bilinear (NHWC, S8)\0"
  "Resize Bilinear (NHWC, U8)\0"
  "Sigmoid (NC, F16)\0"
  "Sigmoid (NC, F32)\0"
  "Sigmoid (NC, QS8)\0"
  "Sigmoid (NC, QU8)\0"
  "Slice (ND, X8)\0"
  "Slice (ND, X16)\0"
  "Slice (ND, X32)\0"
  "Softmax (NC, F16)\0"
  "Softmax (NC, F32)\0"
  "Softmax (NC, QU8)\0"
  "Space To Depth (NHWC, X8)\0"
  "Space To Depth (NHWC, X16)\0"
  "Space To Depth (NHWC, X32)\0"
  "Square (NC, F16)\0"
  "Square (NC, F32)\0"
  "Square Root (NC, F16)\0"
  "Square Root (NC, F32)\0"
  "Squared Difference (NC, F16)\0"
  "Squared Difference (NC, F32)\0"
  "Subtract (ND, F16)\0"
  "Subtract (ND, F32)\0"
  "Subtract (ND, QS8)\0"
  "Subtract (ND, QU8)\0"
  "Tanh (NC, F16)\0"
  "Tanh (NC, F32)\0"
  "Tanh (NC, QS8)\0"
  "Tanh (NC, QU8)\0"
  "Transpose (ND, X8)\0"
  "Transpose (ND, X16)\0"
  "Transpose (ND, X32)\0"
  "Truncation (NC, F16)\0"
  "Truncation (NC, F32)\0"
  "Unpooling (NHWC, X32)";

const char* xnn_operator_type_to_string(enum xnn_operator_type operator_type) {
  assert(operator_type >= xnn_operator_type_invalid);
  assert(operator_type <= xnn_operator_type_unpooling_nhwc_x32);
  return &data[offset[operator_type]];
}
