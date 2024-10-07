/*
template <typename TypeA, typename TypeB>
struct MixedGemmArchTraits<TypeA, TypeB, cutlass::arch::Sm80,
    typename cutlass::platform::enable_if<cutlass::platform::is_same<TypeA, cutlass::half_t>::value
        || cutlass::platform::is_same<TypeA, cutlass::bfloat16_t>::value>::type>
{
private:
    using LayoutDetails = LayoutDetailsB<TypeA, TypeB, cutlass::arch::Sm80>;

public:
    static constexpr int ThreadblockK = LayoutDetails::ThreadblockK;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using AccType = float;
    using LayoutB = typename LayoutDetails::Layout;

    static constexpr int ElementsPerAccessA = 128 / cutlass::sizeof_bits<TypeA>::value;
    static constexpr int ElementsPerAccessB = LayoutDetails::ElementsPerAccess;
    static constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<TypeA>::value;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using Operator = typename LayoutDetails::Operator;
};
*/

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void generic_mixed_gemm_kernelLauncher(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{ 
    using CutlassActivationType = cutlass::float_e4m3_t;    // __nv_fp8_e4m3
    using CutlassWeightType = cutlass::uint4b_t;
    using CutlassScaleZeroType = half;
    using CutlassBiasType = half;
    using CutlassOutputType = half;
    // cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits
        = cutlass::gemm::kernel::MixedGemmArchTraits<CutlassActivationType, CutlassWeightType, arch>;
    using ElementAccumulator = typename MixedGemmArchTraits::AccType;

    constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;
    using EpilogueOp =
        typename tkc::Epilogue<CutlassOutputType, ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    using Operator = typename MixedGemmArchTraits::Operator;
    using TaggedOperator = typename cutlass::arch::TagOperator<Operator, QuantOp>::TaggedOperator;

    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
    CutlassActivationType, cutlass::layout::RowMajor, MixedGemmArchTraits::ElementsPerAccessA, 
    CutlassWeightType, typename MixedGemmArchTraits::LayoutB, MixedGemmArchTraits::ElementsPerAccessB, 
    CutlassOutputType, cutlass::layout::RowMajor, ElementAccumulator,
        cutlass::arch::OpClassTensorOp, arch, ThreadblockShape, WarpShape,
        typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
        typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, true,
        TaggedOperator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch, // Ensure top level arch is used for dispatch
        GemmKernel_::kSplitKSerial>;

    if (occupancy != nullptr)
    {
        *occupancy = tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

    int const ldb = cutlass::platform::is_same<cutlass::layout::RowMajor, typename MixedGemmArchTraits::LayoutB>::value
        ? n
        : k * GemmKernel::kInterleave;

    if (weight_scales == nullptr)
    {
        throw std::runtime_error("Weight scales must always be set to a non-null value.");
    }
    if (group_size != 128)
    {
        throw std::runtime_error("Only group size 128 supported for fine grained W4A(fp)8 kernels.");
    }
    if (weight_zero_points == nullptr)
    {
        throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
    }
   
    int const ld_scale_zero = cutlass::isFinegrained(QuantOp) ? n : 0;
    ElementAccumulator output_op_beta = (biases == nullptr) ? ElementAccumulator(0.f) : ElementAccumulator(1.f);
    typename Gemm::Arguments args({m, n, k}, group_size,
        {reinterpret_cast<CutlassActivationType*>(const_cast<ActivationType*>(A)), k},
        {reinterpret_cast<CutlassWeightType*>(const_cast<WeightType*>(B)), ldb},
        {reinterpret_cast<CutlassScaleZeroType*>(const_cast<ScaleZeroType*>(weight_scales)), ld_scale_zero},
        {reinterpret_cast<CutlassScaleZeroType*>(const_cast<ScaleZeroType*>(weight_zero_points)), ld_scale_zero},
        {reinterpret_cast<CutlassBiasType*>(const_cast<BiasType*>(biases)), 0},
        {reinterpret_cast<CutlassOutputType*>(C), n}, gemm_config.split_k_factor,
        {ElementAccumulator(alpha), output_op_beta});

    Gemm gemm;
    auto can_implement = gemm.can_implement(args);
    auto init_status = gemm.initialize(args, workspace, stream);
    auto run_status = gemm.run(stream);
}