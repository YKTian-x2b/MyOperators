#include "common.h"

#include "cutlass/gemm/device/gemm.h"


cudaError_t CutlassSgemmTT(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc)
{
    using RowMajor = cutlass::layout::RowMajor;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    RowMajor,  // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    RowMajor,  // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    RowMajor>; // Layout of C matrix


    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {A, lda},       // Tensor-ref for source matrix A
                                {B, ldb},       // Tensor-ref for source matrix B
                                {C, ldc},       // Tensor-ref for source matrix C
                                {C, ldc},       // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    cutlass::Status status = gemm_operator(args);
}

int main(int argc, const char *arg[])
{
    int M = 256;
    int N = 512;
    int K = 1024;
    float alpha = 1.0;
    float beta = 0.;

    // 行主序的ld是列数
    int lda = K;
    int ldb = N;
    int ldc = K;

    size_t sizeof_C = sizeof(float) * ldc * M;

    float *A;
    float *B;
    float *C_cutlass;
    float *C_reference;
    GpuAllocateMatrix(&A, M, K, 0);
    GpuAllocateMatrix(&B, K, N, 0);
    GpuAllocateMatrix(&C_cutlass, M, N, 0);
    GpuAllocateMatrix(&C_reference, M, N, 0);

    CutlassSgemmTT(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);
    // ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc);

    std::vector<float> host_cutlass(ldc * N, 0);
    std::vector<float> host_reference(ldc * N, 0);

    CUDA_CHECK(cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(C_reference));
    CUDA_CHECK(cudaFree(C_cutlass));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(A));
}