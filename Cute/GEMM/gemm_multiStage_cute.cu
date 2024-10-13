#include "../common.h"
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

template <typename Config, unsigned M, unsigned N, unsigned K>
__global__ void gemm_multi_stage(void *Dptr, void *Aptr, void *Bptr)
{
    using namespace cute;

    using T = typename Config::T;
    using TiledMMA = typename Config::MMA;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using SmemLayoutC = typename Config::SmemLayoutC;
    using G2SCopyA = typename Config::G2SCopyA;
    using G2SCopyB = typename Config::G2SCopyB;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    // epilogue
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    using S2GCopyC = typename Config::S2GCopyC;

    constexpr unsigned kTileM = Config::kTileM;
    constexpr unsigned kTileN = Config::kTileN;
    constexpr unsigned kTileK = Config::kTileK;
    constexpr unsigned kStage = Config::kStage;

    int tid = threadIdx.x;
    int bix = blockIdx.x;
    int biy = blockIdx.y;

    extern __shared__ T smem[];
    T *Asmem = smem;
    T *Bsmem = smem + cute::cosize(SmemLayoutA{});

    /// 任务分割
    // 给全局内存里的数据创建Tensor
    Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

    // A 按 tiler 切割， 拿到coord坐标的那个子Tensor
    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(biy, _));   // (kTileM, kTileK, k)
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(bix, _));   // (kTileN, kTileK, k)
    Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(biy, bix)); // (kTileM, kTileN)

    Tensor sA = make_tensor(make_smem_ptr(Asmem), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(Bsmem), SmemLayoutB{});

    /// 任务分发
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    // 对逻辑Tensor针对该线程的划分 然后生成对应的寄存器表示
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD);          // (MMA, MMA_M, MMA_N)
    clear(tCrD);

    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

    // 16*16 => 32*32*16
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(tid);
    auto tAsA = s2r_thr_copy_a.partition_S(sA);      // (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(tid);
    auto tBsB = s2r_thr_copy_b.partition_S(sB);     // (CPY, CPY_N, CPY_K, kStage)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    // if (thread0())
    // {
    //     print("  tCrA : ");
    //     print(tCrA);
    //     print("\n");
    //     print("  tAgA_copy : ");
    //     print(tAgA_copy);
    //     print("\n");

    //     print("  tAsA_copy : ");
    //     print(tAsA_copy);
    //     print("\n");
    //     print("  tAsA : ");
    //     print(tAsA);
    //     print("\n");

    //     print("  tCrA_view : ");
    //     print(tCrA_view);
    //     print("\n");
    // }

    int itile_to_read = 0;
    int ismem_write = 0;
    int ismem_read = 0;

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; istage++)
    {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();
        itile_to_read++;
        ismem_write++;
    }

    cp_async_wait<kStage - 2>();
    __syncthreads();

    cute::copy(s2r_tiled_copy_a, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));

    const unsigned ntile = K / kTileK;
    for (int i = 0; i < ntile; i++)
    {
        int nk = size<2>(tCrA);

        /// 下一个tile预取
        if (itile_to_read < ntile)
        {
            cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
            itile_to_read++;
            ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();

        #pragma unroll
        for (int ik = 0; ik < nk; ik++)
        {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1)
            {
                cp_async_wait<kStage - 2>();
                __syncthreads();
                ismem_read = (ismem_read + 1) % kStage;
            }

            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        }
    }

    /// Epilogue
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(tid);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);  // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(tid);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY, CPY_MN)
    int step = size<3>(tCsC_r2s);                 // pipe

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY, CPY_MN)

    // if (thread0())
    // {
    //     print("  tCrC_r2s : ");
    //     print(tCrC_r2s);
    //     print("\n");
    //     print("  tCsC_r2s : ");
    //     print(tCsC_r2s);
    //     print("\n");
    //     print("  tCsC_s2g : ");
    //     print(tCsC_s2g);
    //     print("\n");
    //     print("  tCgC_s2g : ");
    //     print(tCgC_s2g);
    //     print("\n");
    //     print("  tCrC_r2sx : ");
    //     print(tCrC_r2sx);
    //     print("\n");
    // }

#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step)
    {
#pragma unroll
        for (int j = 0; j < step; ++j)
        {
            cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < step; ++j)
        {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    }
}

namespace config
{
    using namespace cute;
    template <typename T_, unsigned kTileM_ = 128, unsigned kTileN_ = 128, unsigned kTileK_ = 32,
              unsigned kStage_ = 5, unsigned kSmemLayoutCBatch_ = 2,
              typename ComputeType = T_>
    struct GemmConfig
    {
        using T = T_;

        static constexpr unsigned kTileM = kTileM_;
        static constexpr unsigned kTileN = kTileN_;
        static constexpr unsigned kTileK = kTileK_;
        static constexpr unsigned kStage = kStage_;
        static constexpr unsigned kSmemLayoutCBatch = kSmemLayoutCBatch_;

        static constexpr int kShmLoadSwizzleB = 3; // 2^B^行
        static constexpr int kShmLoadSwizzleM = 3; // 2^M^个basic基础元素，8*2B=16B
        static constexpr int kShmLoadSwizzleS = 3; // 2^S^列 32*4B/16B=8
        using SmemLayoutAtom = decltype(composition(Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
                                                    make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                                                                make_stride(Int<kTileK>{}, Int<1>{}))));

        // using SmemLayoutAtom = decltype(make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
        //                                             make_stride(Int<kTileK>{}, Int<1>{})));

        using SmemLayoutA = decltype(tile_to_shape(
            SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
        using SmemLayoutB = decltype(tile_to_shape(
            SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

        using mma_op = SM80_16x8x16_F16F16F16F16_TN;
        using mma_traits = MMA_Traits<mma_op>;
        using mma_atom = MMA_Atom<mma_traits>;
        using mma_atom_shape = mma_traits::Shape_MNK;
        static constexpr unsigned kMmaThrLayoutM = 2;
        static constexpr unsigned kMmaThrLayoutN = 2;
        static constexpr unsigned kMmaThrLayoutK = 1;
        static constexpr unsigned kMmaPM = 1 * kMmaThrLayoutM * get<0>(mma_atom_shape{});
        static constexpr unsigned kMmaPN = 2 * kMmaThrLayoutN * get<1>(mma_atom_shape{});
        static constexpr unsigned kMmaPK = 1 * kMmaThrLayoutK * get<2>(mma_atom_shape{});

        using MMA = decltype(make_tiled_mma(
            mma_atom{},
            make_layout(make_shape(Int<kMmaThrLayoutM>{}, Int<kMmaThrLayoutN>{}, Int<kMmaThrLayoutK>{})),
            Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>{}));

        using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
        using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
        using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

        using G2SCopyA = decltype(make_tiled_copy(
            g2s_copy_atom{},
            make_layout(make_shape(Int<32>{}, Int<4>{}),
                        make_stride(Int<4>{}, Int<1>{})),
            make_layout(make_shape(Int<1>{}, Int<8>{}))));
        using G2SCopyB = G2SCopyA;

        using s2r_copy_op = SM75_U32x4_LDSM_N;
        using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
        using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;

        using S2RCopyAtomA = s2r_copy_atom;
        using S2RCopyAtomB = s2r_copy_atom;

        // epilogue
        using SmemLayoutAtomC = decltype(composition(Swizzle<2, 3, 3>{},
                                                     make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                                                                 make_stride(Int<kMmaPN>{}, Int<1>{}))));
        // using SmemLayoutAtomC = decltype(make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
        //                                             make_stride(Int<kMmaPN>{}, Int<1>{})));

        using SmemLayoutC = decltype(tile_to_shape(
            SmemLayoutAtomC{},
            make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

        // SmemA的空间复用
        static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >=
                          size(SmemLayoutC{}),
                      "C shared memory request is large than A's one pipe");

        // dst = static_cast<D>(static_cast<S>(src));
        using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

        using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
        using S2GCopyC =
            decltype(make_tiled_copy(S2GCopyAtomC{},
                                     make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                 make_stride(Int<4>{}, Int<1>{})),
                                     make_layout(make_shape(Int<1>{}, Int<8>{}))));

        static constexpr unsigned ThdsPerBLK = size(MMA{});
        static constexpr unsigned shm_size_AB = cosize(SmemLayoutA{}) + cosize(SmemLayoutB{});
        static constexpr unsigned shm_size_C = cosize(SmemLayoutC{});
        static constexpr unsigned kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);
    };

} // namespace config

int main()
{
    using T = cute::half_t;
    using namespace cute;

    constexpr unsigned M = 81920;
    constexpr unsigned N = 256;
    constexpr unsigned K = 256;
    constexpr unsigned MN = M * N;
    constexpr unsigned MK = M * K;
    constexpr unsigned NK = N * K;

    thrust::host_vector<T> Aptr_h(MK);
    thrust::host_vector<T> Bptr_h(NK);
    thrust::host_vector<T> Dptr_h(MN);
    for(int i = 0; i < MK; i++){
        Aptr_h[i] = static_cast<T>(2*(rand() / double(RAND_MAX)) - 1);
    }
    for(int i = 0; i < NK; i++){
        Bptr_h[i] = static_cast<T>(2*(rand() / double(RAND_MAX)) - 1);
    }
    for(int i = 0; i < MN; i++){
        Dptr_h[i] = 0.;
    }
    thrust::device_vector<T> Aptr_d = Aptr_h;
    thrust::device_vector<T> Bptr_d = Bptr_h;
    thrust::device_vector<T> Dptr_d = Dptr_h;
    thrust::device_vector<T> Dptr_cublas_d = Dptr_h;

    /// my
    config::GemmConfig<T, 128, 128, 32, 3> gemm_config;
    // print(typename decltype(gemm_config)::MMA{});
    // print(typename decltype(gemm_config)::SmemLayoutA{});

    constexpr unsigned blk_size = gemm_config.ThdsPerBLK;
    dim3 grd_size(ceil_div(N, gemm_config.kTileN), ceil_div(M, gemm_config.kTileM));
    constexpr unsigned smem_size = gemm_config.kShmSize;
    cudaFuncSetAttribute(gemm_multi_stage<decltype(gemm_config), M, N, K>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    gemm_multi_stage<decltype(gemm_config), M, N, K>
        <<<grd_size, blk_size, smem_size>>>(Dptr_d.data().get(), 
                                            Aptr_d.data().get(), 
                                            Bptr_d.data().get());
    CUDA_CHECK(cudaDeviceSynchronize());
    Dptr_h = Dptr_d;

    /// cublas
    const half alpha = 1.0;
    const half beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                (half *)Bptr_d.data().get(), K, 
                (half *)Aptr_d.data().get(), K, &beta, 
                (half *)Dptr_cublas_d.data().get(), N);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::host_vector<T> Dptr_cublas_h = Dptr_cublas_d;
    cublasDestroy(handle);

    /// valid
    CheckRes(Dptr_h.data(), Dptr_cublas_h.data(), M, N, 0.01);
}

