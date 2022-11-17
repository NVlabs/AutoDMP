/*
* SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: Apache-2.0
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/**
 * @file   rudy_cuda.cpp
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 * A routing/pin utilization estimator based on the following two papers
 * "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng,
 * ICCAD'94
 * Anthony Agnesina: added macro blockages effect on congestion
 */

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

// fill the demand map net by net
template <typename T>
int rudyNetsCudaLauncher(const T *pin_pos_x, const T *pin_pos_y,
                         const int *netpin_start, const int *flat_netpin,
                         const T *net_weights, T bin_size_x, T bin_size_y, T xl,
                         T yl, T xh, T yh, int num_bins_x, int num_bins_y,
                         int num_nets, T *horizontal_utilization_map,
                         T *vertical_utilization_map);

// fill the demand map of macros
template <typename T>
int rudyMacrosCudaLauncher(const T *macro_pos_x, const T *macro_pos_y,
                           const T *macro_size_x, const T *macro_size_y,
                           const T *unit_macro_util_H,
                           const T *unit_macro_util_V, T bin_size_x,
                           T bin_size_y, T xl, T yl, T xh, T yh, int num_bins_x,
                           int num_bins_y, int num_macros,
                           T *horizontal_utilization_map,
                           T *vertical_utilization_map);

void rudy_nets_forward(at::Tensor pin_pos, at::Tensor netpin_start,
                       at::Tensor flat_netpin, at::Tensor net_weights,
                       double bin_size_x, double bin_size_y, double xl,
                       double yl, double xh, double yh, int num_bins_x,
                       int num_bins_y, at::Tensor horizontal_utilization_map,
                       at::Tensor vertical_utilization_map) {
  CHECK_FLAT_CUDA(pin_pos);
  CHECK_EVEN(pin_pos);
  CHECK_CONTIGUOUS(pin_pos);

  CHECK_FLAT_CUDA(netpin_start);
  CHECK_CONTIGUOUS(netpin_start);

  CHECK_FLAT_CUDA(flat_netpin);
  CHECK_CONTIGUOUS(flat_netpin);

  CHECK_FLAT_CUDA(net_weights);
  CHECK_CONTIGUOUS(net_weights);

  int num_nets = netpin_start.numel() - 1;
  int num_pins = pin_pos.numel() / 2;

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pin_pos, "rudyNetsCudaLauncher", [&] {
    rudyNetsCudaLauncher<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(pin_pos, scalar_t) + num_pins,
        DREAMPLACE_TENSOR_DATA_PTR(netpin_start, int),
        DREAMPLACE_TENSOR_DATA_PTR(flat_netpin, int),
        (net_weights.numel())
            ? DREAMPLACE_TENSOR_DATA_PTR(net_weights, scalar_t)
            : nullptr,
        bin_size_x, bin_size_y, xl, yl, xh, yh, num_bins_x, num_bins_y,
        num_nets,
        DREAMPLACE_TENSOR_DATA_PTR(horizontal_utilization_map, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(vertical_utilization_map, scalar_t));
  });
}

void rudy_macros_forward(at::Tensor macro_pos_x, at::Tensor macro_pos_y,
                         at::Tensor macro_size_x, at::Tensor macro_size_y,
                         at::Tensor unit_macro_util_H,
                         at::Tensor unit_macro_util_V, double bin_size_x,
                         double bin_size_y, double xl, double yl, double xh,
                         double yh, int num_bins_x, int num_bins_y,
                         at::Tensor horizontal_utilization_map,
                         at::Tensor vertical_utilization_map) {
  // assume these tensors are all built the same way
  CHECK_FLAT_CUDA(macro_pos_x);
  CHECK_CONTIGUOUS(macro_pos_x);

  int num_macros = macro_pos_x.numel();

  // Call the cuda kernel launcher
  DREAMPLACE_DISPATCH_FLOATING_TYPES(
      macro_pos_x, "rudyMacrosCudaLauncher", [&] {
        rudyMacrosCudaLauncher<scalar_t>(
            DREAMPLACE_TENSOR_DATA_PTR(macro_pos_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(macro_pos_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(macro_size_x, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(macro_size_y, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(unit_macro_util_H, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(unit_macro_util_V, scalar_t), bin_size_x,
            bin_size_y, xl, yl, xh, yh, num_bins_x, num_bins_y, num_macros,
            DREAMPLACE_TENSOR_DATA_PTR(horizontal_utilization_map, scalar_t),
            DREAMPLACE_TENSOR_DATA_PTR(vertical_utilization_map, scalar_t));
      });
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nets_forward", &DREAMPLACE_NAMESPACE::rudy_nets_forward,
        "compute RUDY map (CUDA)");
  m.def("macros_forward", &DREAMPLACE_NAMESPACE::rudy_macros_forward,
        "compute RUDY map of macros (CUDA");
}


