/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

// #include <limits>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"

#include "cutlass/trace.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace device
{

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmKernel_>
class GemmUniversalBaseCompat
{
public:
    using GemmKernel = GemmKernel_;
    using ThreadblockShape = typename GemmKernel::Mma::Shape;

    using ElementA = typename GemmKernel::ElementA;
    using LayoutA = typename GemmKernel::LayoutA;
    using TensorRefA = TensorRef<ElementA const, LayoutA>;
    static ComplexTransform const kTransformA = GemmKernel::kTransformA;

    using ElementB = typename GemmKernel::ElementB;
    using LayoutB = typename GemmKernel::LayoutB;
    using TensorRefB = TensorRef<ElementB const, LayoutB>;
    static ComplexTransform const kTransformB = GemmKernel::kTransformB;

    using ElementC = typename GemmKernel::ElementC;
    using LayoutC = typename GemmKernel::LayoutC;
    using TensorRefC = TensorRef<ElementC const, LayoutC>;
    using TensorRefD = TensorRef<ElementC, LayoutC>;

    using ElementAccumulator = typename GemmKernel::Mma::Policy::Operator::ElementC;

    using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
    using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
    using Operator = typename GemmKernel::Operator;

    using Arguments = typename GemmKernel::Arguments;

protected:
    typename GemmKernel::Params params_;
    // {M/kM, N/kN, args.batch_count} grid.z向上取整 gemm_k_size保证是128B/size(T)的倍数。
    static void get_grid_shape_(gemm::GemmCoord& grid_tiled_shape, int& gemm_k_size, Arguments const& args)
    {
        ThreadblockSwizzle threadblock_swizzle;
        // 就是常规的 {M/kM, N/kN, batch}
        grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
            args.problem_size, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, args.batch_count);
        
        gemm_k_size = args.problem_size.k();

        if (args.mode == GemmUniversalMode::kGemm || args.mode == GemmUniversalMode::kGemmSplitKParallel)
        {
            // 128字节对齐
            int const kAlignK
                = const_max(const_max(128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value), 1);

            // 上取整为kAlignK的倍数
            gemm_k_size = round_up(ceil_div(args.problem_size.k(), args.batch_count), kAlignK);

            if (gemm_k_size)
            {
                grid_tiled_shape.k() = ceil_div(args.problem_size.k(), gemm_k_size);
            }
        }
    }

public:
    /// Constructs the GEMM.
    GemmUniversalBaseCompat() {}

    /// Determines whether the GEMM can execute the given problem.
    static Status can_implement(Arguments const& args)
    {

    }

    /// Gets the workspace size
    static size_t get_workspace_size(Arguments const& args)
    {

    }

    /// Computes the grid shape
    static dim3 get_grid_shape(Arguments const& args)
    {

    }

    /// Computes the maximum number of active blocks per multiprocessor
    static int maximum_active_blocks(int smem_capacity = -1)
    {

    }

    /// Initializes GEMM state from arguments.
    Status initialize(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr)
    {
        size_t workspace_bytes = get_workspace_size(args);
        if (workspace_bytes)
        {

            if (!workspace)
            {
                return Status::kErrorWorkspaceNull;
            }

            if (args.mode == GemmUniversalMode::kGemm)
            {
                cudaMemsetAsync(workspace, 0, workspace_bytes, stream);
            }
        }
        cutlass::gemm::GemmCoord grid_tiled_shape;
        int gemm_k_size = 0;
        get_grid_shape_(grid_tiled_shape, gemm_k_size, args);

        // Initialize the Params structure
        params_ = typename GemmKernel::Params(args, grid_tiled_shape, gemm_k_size, static_cast<int*>(workspace));

        // Specify shared memory capacity for kernel.
        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));

        if (smem_size >= (48 << 10))
        {
            cudaError_t result
                = cudaFuncSetAttribute(Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess)
            {
                return Status::kErrorInternal;
            }
        }

        return Status::kSuccess;
    }

    /// Lightweight update given a subset of arguments
    Status update(Arguments const& args, void* workspace = nullptr)
    {

    }

    /// Runs the kernel using initialized state.
    Status run(cudaStream_t stream = nullptr)
    {
        ThreadblockSwizzle threadblock_swizzle;
        // 根据逻辑grid_tiled_shape计算 swizzle后的实际启动Grid
        dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(GemmKernel::kThreadCount, 1, 1);

        int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
        cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params_);
    }

    /// Runs the kernel using initialized state.
    Status operator()(cudaStream_t stream = nullptr)
    {
        return run(stream);
    }

    /// Runs the kernel using initialized state.
    Status operator()(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr)
    {
        initialize(args, workspace, stream);
        return run(stream);
    }
};

} // namespace device
} // namespace gemm
} // namespace cutlass
