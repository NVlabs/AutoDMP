/*
* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @file   rudy_cuda_kernel.cu
 * @author Zixuan Jiang, Jiaqi Gu, Yibo Lin
 * @date   Dec 2019
 * @brief  Compute the RUDY/RISA map for routing demand.
 * A routing/pin utilization estimator based on the following two papers
 * "Fast and Accurate Routing Demand Estimation for efficient
 * Routability-driven Placement", by Peter Spindler, DATE'07 "RISA: Accurate and
 * Efficient Placement Routability Modeling", by Chih-liang Eric Cheng, ICCAD'94
 */

 #include "rudy/src/parameters.h"
 #include "utility/src/utils.cuh"
 
 DREAMPLACE_BEGIN_NAMESPACE
 
 template <typename T>
 inline __device__ DEFINE_NET_WIRING_DISTRIBUTION_MAP_WEIGHT;
 
 template <typename T>
 __global__ void rudyNets(const T *pin_pos_x, const T *pin_pos_y,
                          const int *netpin_start, const int *flat_netpin,
                          const T *net_weights, T bin_size_x, T bin_size_y, T xl,
                          T yl, T xh, T yh, int num_bins_x, int num_bins_y,
                          int num_nets, T *horizontal_utilization_map,
                          T *vertical_utilization_map) {
   const int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i < num_nets) {
     const int start = netpin_start[i];
     const int end = netpin_start[i + 1];
 
     T x_max = -cuda::numeric_limits<T>::max();
     T x_min = cuda::numeric_limits<T>::max();
     T y_max = -cuda::numeric_limits<T>::max();
     T y_min = cuda::numeric_limits<T>::max();
 
     for (int j = start; j < end; ++j) {
       int pin_id = flat_netpin[j];
       const T xx = pin_pos_x[pin_id];
       x_max = DREAMPLACE_STD_NAMESPACE::max(xx, x_max);
       x_min = DREAMPLACE_STD_NAMESPACE::min(xx, x_min);
       const T yy = pin_pos_y[pin_id];
       y_max = DREAMPLACE_STD_NAMESPACE::max(yy, y_max);
       y_min = DREAMPLACE_STD_NAMESPACE::min(yy, y_min);
     }
 
     // compute the bin box that this net will affect
     int bin_index_xl = int((x_min - xl) / bin_size_x);
     int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
     bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
     bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);
 
     int bin_index_yl = int((y_min - yl) / bin_size_y);
     int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
     bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
     bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);
 
     T wt = netWiringDistributionMapWeight<T>(end - start);
     if (net_weights) {
       wt *= net_weights[i];
     }
 
     for (int x = bin_index_xl; x < bin_index_xh; ++x) {
       for (int y = bin_index_yl; y < bin_index_yh; ++y) {
         T bin_xl = xl + x * bin_size_x;
         T bin_yl = yl + y * bin_size_y;
         T bin_xh = bin_xl + bin_size_x;
         T bin_yh = bin_yl + bin_size_y;
         T overlap = DREAMPLACE_STD_NAMESPACE::max(
                         DREAMPLACE_STD_NAMESPACE::min(x_max, bin_xh) -
                             DREAMPLACE_STD_NAMESPACE::max(x_min, bin_xl),
                         (T)0) *
                     DREAMPLACE_STD_NAMESPACE::max(
                         DREAMPLACE_STD_NAMESPACE::min(y_max, bin_yh) -
                             DREAMPLACE_STD_NAMESPACE::max(y_min, bin_yl),
                         (T)0);
         overlap *= wt;
         int index = x * num_bins_y + y;
         // Following Wuxi's implementation, a tolerance is added to avoid 0-size
         // bounding box
         atomicAdd(
             horizontal_utilization_map + index,
             overlap / (y_max - y_min + cuda::numeric_limits<T>::epsilon()));
         atomicAdd(
             vertical_utilization_map + index,
             overlap / (x_max - x_min + cuda::numeric_limits<T>::epsilon()));
       }
     }
   }
 }
 
 template <typename T>
 __global__ void rudyMacros(const T *macro_pos_x, const T *macro_pos_y,
                            const T *macro_size_x, const T *macro_size_y,
                            const T *unit_macro_util_H,
                            const T *unit_macro_util_V, T bin_size_x,
                            T bin_size_y, T xl, T yl, T xh, T yh, int num_bins_x,
                            int num_bins_y, int num_macros,
                            T *horizontal_utilization_map,
                            T *vertical_utilization_map) {
   const int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i < num_macros) {
     T util_H = unit_macro_util_H[i];
     T util_V = unit_macro_util_V[i];
 
     T x_min = macro_pos_x[i];
     T x_max = x_min + macro_size_x[i];
     T y_min = macro_pos_y[i];
     T y_max = y_min + macro_size_y[i];
 
     int bin_index_xl = int((x_min - xl) / bin_size_x);
     int bin_index_xh = int((x_max - xl) / bin_size_x) + 1;
     bin_index_xl = DREAMPLACE_STD_NAMESPACE::max(bin_index_xl, 0);
     bin_index_xh = DREAMPLACE_STD_NAMESPACE::min(bin_index_xh, num_bins_x);
 
     int bin_index_yl = int((y_min - yl) / bin_size_y);
     int bin_index_yh = int((y_max - yl) / bin_size_y) + 1;
     bin_index_yl = DREAMPLACE_STD_NAMESPACE::max(bin_index_yl, 0);
     bin_index_yh = DREAMPLACE_STD_NAMESPACE::min(bin_index_yh, num_bins_y);
 
     for (int x = bin_index_xl; x < bin_index_xh; ++x) {
       for (int y = bin_index_yl; y < bin_index_yh; ++y) {
         T bin_xl = xl + x * bin_size_x;
         T bin_yl = yl + y * bin_size_y;
         T bin_xh = bin_xl + bin_size_x;
         T bin_yh = bin_yl + bin_size_y;
         T overlap = DREAMPLACE_STD_NAMESPACE::max(
                         DREAMPLACE_STD_NAMESPACE::min(x_max, bin_xh) -
                             DREAMPLACE_STD_NAMESPACE::max(x_min, bin_xl),
                         (T)0) *
                     DREAMPLACE_STD_NAMESPACE::max(
                         DREAMPLACE_STD_NAMESPACE::min(y_max, bin_yh) -
                             DREAMPLACE_STD_NAMESPACE::max(y_min, bin_yl),
                         (T)0);
         T overlap_h = overlap * util_H;
         T overlap_v = overlap * util_V;
         int index = x * num_bins_y + y;
         atomicAdd(horizontal_utilization_map + index, overlap_h);
         atomicAdd(vertical_utilization_map + index, overlap_v);
       }
     }
   }
 }
 
 // fill the demand map net by net
 template <typename T>
 int rudyNetsCudaLauncher(const T *pin_pos_x, const T *pin_pos_y,
                          const int *netpin_start, const int *flat_netpin,
                          const T *net_weights, T bin_size_x, T bin_size_y, T xl,
                          T yl, T xh, T yh, int num_bins_x, int num_bins_y,
                          int num_nets, T *horizontal_utilization_map,
                          T *vertical_utilization_map) {
   int thread_count = 512;
   int block_count = ceilDiv(num_nets, thread_count);
   rudyNets<<<block_count, thread_count>>>(
       pin_pos_x, pin_pos_y, netpin_start, flat_netpin, net_weights, bin_size_x,
       bin_size_y, xl, yl, xh, yh, num_bins_x, num_bins_y, num_nets,
       horizontal_utilization_map, vertical_utilization_map);
   return 0;
 }
 
 // fill the demand map of macros
 template <typename T>
 int rudyMacrosCudaLauncher(const T *macro_pos_x, const T *macro_pos_y,
                            const T *macro_size_x, const T *macro_size_y,
                            const T *unit_macro_util_H,
                            const T *unit_macro_util_V, T bin_size_x,
                            T bin_size_y, T xl, T yl, T xh, T yh, int num_bins_x,
                            int num_bins_y, int num_macros,
                            T *horizontal_utilization_map,
                            T *vertical_utilization_map) {
   int thread_count = 512;
   int block_count = ceilDiv(num_macros, thread_count);
   rudyMacros<<<block_count, thread_count>>>(
       macro_pos_x, macro_pos_y, macro_size_x, macro_size_y, unit_macro_util_H,
       unit_macro_util_V, bin_size_x, bin_size_y, xl, yl, xh, yh, num_bins_x,
       num_bins_y, num_macros, horizontal_utilization_map,
       vertical_utilization_map);
   return 0;
 }
 
 #define REGISTER_NETS_KERNEL_LAUNCHER(T)                                    \
   template int rudyNetsCudaLauncher<T>(                                     \
       const T *pin_pos_x, const T *pin_pos_y, const int *netpin_start,      \
       const int *flat_netpin, const T *net_weights, T bin_size_x,           \
       T bin_size_y, T xl, T yl, T xh, T yh, int num_bins_x, int num_bins_y, \
       int num_nets, T *horizontal_utilization_map,                          \
       T *vertical_utilization_map);
 
 #define REGISTER_MACROS_KERNEL_LAUNCHER(T)                                \
   template int rudyMacrosCudaLauncher<T>(                                 \
       const T *macro_pos_x, const T *macro_pos_y, const T *macro_size_x,  \
       const T *macro_size_y, const T *unit_macro_util_H,                  \
       const T *unit_macro_util_V, T bin_size_x, T bin_size_y, T xl, T yl, \
       T xh, T yh, int num_bins_x, int num_bins_y, int num_macros,         \
       T *horizontal_utilization_map, T *vertical_utilization_map);
 
 REGISTER_NETS_KERNEL_LAUNCHER(float);
 REGISTER_NETS_KERNEL_LAUNCHER(double);
 
 REGISTER_MACROS_KERNEL_LAUNCHER(float);
 REGISTER_MACROS_KERNEL_LAUNCHER(double);
 
 DREAMPLACE_END_NAMESPACE
 
 
 