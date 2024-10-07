#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/arch/arch.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include <type_traits>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass
{
namespace gemm
{
namespace kernel
{

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail
{
template <typename>
inline constexpr bool dependent_false_v = false;
}

template <typename Mma_,          ///! Threadblock-scoped matrix multiply-accumulate
    typename Epilogue_,           ///! Epilogue
    typename ThreadblockSwizzle_, ///! Threadblock swizzling function
    typename KernelArch, ///! The Architecture this kernel is compiled for. Used since SIMT kernels lose top-level
                         /// arch.
    bool SplitKSerial    ///! If true, code supporting split-K via serial reduction is enabled.
    >
struct GemmFpAIntB
{

    using Mma = Mma_;
    using Epilogue = Epilogue_;
    using EpilogueOutputOp = typename Epilogue::OutputOp;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    static bool const kSplitKSerial = SplitKSerial;

    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB = typename Mma::IteratorB::Element;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC = typename Mma::LayoutC;
    using ElementScale = ElementC;

    static ComplexTransform const kTransformA = Mma::kTransformA;
    static ComplexTransform const kTransformB = Mma::kTransformA;

    // Type definitions about the mainloop.
    using Operator = typename Mma::Operator;
    using OperatorClass = typename Mma::Operator::OperatorClass;
    using ThreadblockShape = typename Mma::Shape;
    using WarpShape = typename Mma::Operator::Shape;
    using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
    using ArchTag = typename Mma::ArchTag;

    static int const kStages = Mma::kStages;
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    /// Warp count (concept: GemmShape)
    using WarpCount = typename Mma::WarpCount;
    static int const kThreadCount = 32 * WarpCount::kCount;

    static constexpr int kInterleave = Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

    /// Parameters structure
    struct Arguments
    {
        GemmUniversalMode mode = GemmUniversalMode::kGemm;

        cutlass::gemm::GemmCoord problem_size;
        int group_size;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Mma::IteratorScale::TensorRef ref_scale;
        typename Mma::IteratorScale::TensorRef ref_zero;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;

        // Control serial split-k
        int batch_count;

        typename EpilogueOutputOp::Params output_op;

        // For gather+scatter operations
        int const* gather_A_indices;
        int const* gather_B_indices;
        int const* scatter_D_indices;

        // Included so we can use Gemm Universal
        int batch_stride_D = 0;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Arguments() {}

        CUTLASS_HOST_DEVICE
        Arguments(cutlass::gemm::GemmCoord const& problem_size, int const group_size,
            typename Mma::IteratorA::TensorRef ref_A, typename Mma::IteratorB::TensorRef ref_B,
            typename Mma::IteratorScale::TensorRef ref_scale, typename Mma::IteratorScale::TensorRef ref_zero,
            typename Epilogue::OutputTileIterator::TensorRef ref_C,
            typename Epilogue::OutputTileIterator::TensorRef ref_D, int serial_split_k_factor,
            typename EpilogueOutputOp::Params output_op = typename EpilogueOutputOp::Params(),
            int const* gather_A_indices = nullptr, int const* gather_B_indices = nullptr,
            int const* scatter_D_indices = nullptr)
            : problem_size(problem_size)
            , group_size(group_size)
            , ref_A(ref_A)
            , ref_B(ref_B)
            , ref_scale(ref_scale)
            , ref_zero(ref_zero)
            , ref_C(ref_C)
            , ref_D(ref_D)
            , batch_count(serial_split_k_factor)
            , output_op(output_op)
            , gather_A_indices(gather_A_indices)
            , gather_B_indices(gather_B_indices)
            , scatter_D_indices(scatter_D_indices)
        {
        }
    };

    /// Parameters structure
    struct Params
    {
        cutlass::gemm::GemmCoord problem_size;
        int group_size;
        cutlass::gemm::GemmCoord grid_tiled_shape;
        int swizzle_log_tile;
        typename Mma::IteratorA::Params params_A;
        typename Mma::IteratorA::TensorRef ref_A;
        typename Mma::IteratorB::Params params_B;
        typename Mma::IteratorB::TensorRef ref_B;
        typename Mma::IteratorScale::Params params_scale;
        typename Mma::IteratorScale::TensorRef ref_scale;
        typename Mma::IteratorScale::TensorRef ref_zero;
        typename Epilogue::OutputTileIterator::Params params_C;
        typename Epilogue::OutputTileIterator::TensorRef ref_C;
        typename Epilogue::OutputTileIterator::Params params_D;
        typename Epilogue::OutputTileIterator::TensorRef ref_D;
        typename EpilogueOutputOp::Params output_op;
        int* semaphore;
        int gemm_k_size;
        // For gather+scatter operations
        int const* gather_A_indices;
        int const* gather_B_indices;
        int const* scatter_D_indices;

        //
        // Methods
        //

        CUTLASS_HOST_DEVICE
        Params()
            : swizzle_log_tile(0)
            , semaphore(0)
            , gemm_k_size(0)
        {
        }

        CUTLASS_HOST_DEVICE
        Params(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape, int const gemm_k_size,
            void* workspace = nullptr)
            : problem_size(args.problem_size)
            , group_size(args.group_size)
            , grid_tiled_shape(grid_tiled_shape)
            , swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape))
            , params_A(args.ref_A.layout())
            , ref_A(args.ref_A)
            , params_B(args.ref_B.layout())
            , ref_B(args.ref_B)
            , params_scale(args.ref_scale.layout())
            , ref_scale(args.ref_scale)
            , ref_zero(args.ref_zero)
            , params_C(args.ref_C.layout())
            , ref_C(args.ref_C)
            , params_D(args.ref_D.layout())
            , ref_D(args.ref_D)
            , output_op(args.output_op)
            , semaphore(static_cast<int*>(workspace))
            , gemm_k_size(gemm_k_size)
            , gather_A_indices(args.gather_A_indices)
            , gather_B_indices(args.gather_B_indices)
            , scatter_D_indices(args.scatter_D_indices)
        {
        }
    };

    /// Shared memory storage structure
    union SharedStorage
    {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    GemmFpAIntB() {}

    /// Determines whether kernel satisfies alignment
    static Status can_implement(Arguments const& args)
    {
        
    }

    static size_t get_extra_workspace_size(Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape)
    {

        return 0;
    }

    // fine grained scale+bias iterator
    template <typename IteratorScale, WeightOnlyQuantOp op, std::enable_if_t<isFinegrained(op), bool> = true>
    CUTLASS_DEVICE static IteratorScale initialize_scale(typename IteratorScale::Params const& params,
        typename IteratorScale::Pointer pointer_scale, typename IteratorScale::Pointer pointer_zero,
        typename IteratorScale::TensorCoord extent, int thread_id,
        typename IteratorScale::TensorCoord const& threadblock_offset, int group_size)
    {
        return IteratorScale(params, pointer_scale, pointer_zero, extent, thread_id, threadblock_offset, group_size);
    }

    template <typename IteratorScale, WeightOnlyQuantOp op, std::enable_if_t<!isFinegrained(op), bool> = true>
    CUTLASS_DEVICE static IteratorScale initialize_scale(typename IteratorScale::Params const& params,
        typename IteratorScale::Pointer pointer_scale, typename IteratorScale::Pointer pointer_zero,
        typename IteratorScale::TensorCoord extent, int thread_id,
        typename IteratorScale::TensorCoord const& threadblock_offset, int group_size)
    {

    }

    CUTLASS_DEVICE
    void run_kernel_(Params const& params, SharedStorage& shared_storage)
    {
        ThreadblockSwizzle threadblock_swizzle;
        cutlass::gemm::GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m()
            || params.grid_tiled_shape.n() <= threadblock_tile_offset.n())
        {
            return;
        }

        // tb处理一个mma宽的M 处理split后的一块K(gemm_k_size)
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * Mma::Shape::kM,
            threadblock_tile_offset.k() * params.gemm_k_size,
        };
        // kInterleave是把多tile的行交错进某一tile的行以确保 AB load等量k维元素时 B还能ldg.128
        cutlass::MatrixCoord tb_offset_B{threadblock_tile_offset.k() * params.gemm_k_size * kInterleave,
            threadblock_tile_offset.n() * Mma::Shape::kN / kInterleave};
        // group_size是64？
        typename MatrixCoord::Index fg_row_offset = threadblock_tile_offset.k() * params.gemm_k_size / 64;
        typename MatrixCoord::Index scale_row_offset = isFinegrained(Mma::QuantOp) ? fg_row_offset : 0;
        cutlass::MatrixCoord tb_offset_scale{scale_row_offset, threadblock_tile_offset.n() * Mma::Shape::kN};

        // Problem size is a function of threadblock index in the K dimension
        // tbK_end
        int problem_size_k = min(params.problem_size.k(), (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

        // Compute threadblock-scoped matrix multiply-add
        // tbK_iter
        int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(params.params_A, params.ref_A.data(),
            {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A, params.gather_A_indices);
        // weight就是 {tileK*kInterleave, tileN}
        typename Mma::IteratorB iterator_B(params.params_B, params.ref_B.data(),
            {problem_size_k * kInterleave, params.problem_size.n() / kInterleave}, thread_idx, tb_offset_B,
            params.gather_B_indices);
        // scale就是 {tileK/GS, tileN}
        typename MatrixCoord::Index scale_row_extent = isFinegrained(Mma::QuantOp) ? problem_size_k / 64 : 1;
        typename Mma::IteratorScale iterator_scale = initialize_scale<typename Mma::IteratorScale, Mma::QuantOp>(
            params.params_scale, params.ref_scale.data(), params.ref_zero.data(),
            {scale_row_extent, params.problem_size.n()}, thread_idx, tb_offset_scale, params.group_size);

        int warp_idx = canonical_warp_idx_sync();
        int lane_idx = threadIdx.x % 32;

        Mma mma(shared_storage.main_loop, params.group_size, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;
        accumulators.clear();

        if (!kSplitKSerial || gemm_k_iterations > 0)
        {
            // Compute threadblock-scoped matrix multiply-add
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_scale, accumulators);
        }

        //
        ///TODO: Epilogue
        //

    }

    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        run_kernel_(params, shared_storage);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass
