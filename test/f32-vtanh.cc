// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vtanh.yaml
//   Generator: tools/generate-vunary-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/vunary.h>
#include "vunary-microkernel-tester.h"


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__SCALAR_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X1, batch_gt_1) {
  for (size_t batch_size = 1 + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X1, inplace) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, batch_gt_2) {
  for (size_t batch_size = 2 + 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X2, inplace) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
  for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}

TEST(F32_VTANH__FMA_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params);
  }
}


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE2_EXPM1MINUS_RR1_P6H5_NR2_X16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse2_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR1_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_div_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 4; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 4 + 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X4, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x4, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X8, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_div_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24; batch_size < 120; batch_size += 12) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 12; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 12 + 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X12, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 60; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x12, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X16, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_eq_20) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(20)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_div_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 40; batch_size < 200; batch_size += 20) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_lt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 20; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, batch_gt_20) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 20 + 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X20, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 100; batch_size += 19) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x20, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_SSE41;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_div_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__SSE41_EXPM1MINUS_RR1_P6H5_NR2_X24, inplace) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__sse41_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_LUT4_P4H2_PERM_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_lut4_p4h2_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR1_X80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X8, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X16, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X24, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X32, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X40, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X48, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X56, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X64, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X72, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX_EXPM1MINUS_RR1_P6H5_NR2_X80, inplace) {
    TEST_REQUIRES_X86_AVX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx_expm1minus_rr1_p6h5_nr2_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_DIV_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT8_P4H3_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut8_p4h3_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_eq_40) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_div_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_lt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_gt_40) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_eq_56) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_div_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_lt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_gt_56) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_eq_72) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_div_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_lt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_gt_72) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__FMA3_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_FMA3;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__fma3_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 8 + 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x8, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x16, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48; batch_size < 240; batch_size += 24) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 24; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 24 + 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 120; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x24, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x32, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(40)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80; batch_size < 400; batch_size += 40) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 40; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, batch_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 40 + 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X40, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 200; batch_size += 39) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x40, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x48, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_eq_56) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(56)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_div_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 112; batch_size < 560; batch_size += 56) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_lt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 56; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, batch_gt_56) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 56 + 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X56, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 280; batch_size += 55) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x56, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x64, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(72)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 144; batch_size < 720; batch_size += 72) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 72; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, batch_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 72 + 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X72, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 360; batch_size += 71) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x72, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX2_EXPM1MINUS_RR1_P6H5_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx2_expm1minus_rr1_p6h5_nr1adj_x80, xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_DIV_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT4_P4H3_PERM_NR1ADJ_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_DIV_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_GATHER_NR1ADJ_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_gather_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_DIV_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_LUT8_P4H3_PERM_NR1ADJ_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut8_p4h3_perm_nr1adj_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_DIV_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_div_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 16 + 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X16, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x16, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 32 + 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X32, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x32, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96; batch_size < 480; batch_size += 48) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 48; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 48 + 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X48, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 240; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x48, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 64 + 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X64, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x64, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_eq_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(80)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_div_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160; batch_size < 800; batch_size += 80) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_lt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 80; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X80, batch_gt_80) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 80 + 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X80, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 400; batch_size += 79) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x80, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X96, batch_eq_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(96)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X96, batch_div_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 192; batch_size < 960; batch_size += 96) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X96, batch_lt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 96; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X96, batch_gt_96) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 96 + 1; batch_size < 192; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X96, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 480; batch_size += 95) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x96, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X112, batch_eq_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(112)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X112, batch_div_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 224; batch_size < 1120; batch_size += 112) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X112, batch_lt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 112; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X112, batch_gt_112) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 112 + 1; batch_size < 224; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X112, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 560; batch_size += 111) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x112, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X128, batch_eq_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(128)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X128, batch_div_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 256; batch_size < 1280; batch_size += 128) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X128, batch_lt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 128; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X128, batch_gt_128) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 128 + 1; batch_size < 256; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X128, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 640; batch_size += 127) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x128, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X144, batch_eq_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(144)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X144, batch_div_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 288; batch_size < 1440; batch_size += 144) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X144, batch_lt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 144; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X144, batch_gt_144) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 144 + 1; batch_size < 288; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X144, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 720; batch_size += 143) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x144, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X160, batch_eq_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    VUnaryMicrokernelTester()
      .batch_size(160)
      .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X160, batch_div_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 320; batch_size < 1600; batch_size += 160) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X160, batch_lt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size < 160; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X160, batch_gt_160) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 160 + 1; batch_size < 320; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }

  TEST(F32_VTANH__AVX512SKX_EXPM1MINUS_RR1_P6H5_NR1_X160, inplace) {
    TEST_REQUIRES_X86_AVX512SKX;
    for (size_t batch_size = 1; batch_size <= 800; batch_size += 159) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_p6h5_nr1_x160, xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
