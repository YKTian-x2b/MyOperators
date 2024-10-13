set -e

# nvcc -G -g --generate-code arch=compute_86,code=sm_86 --expt-relaxed-constexpr -I/usr/local/cuda/include -I/home/yujixuan/kaiPro/cute-gemm-main/3rd/cutlass/include -L/usr/local/cuda/lib64 -l cuda -l cublas -o res/gemm_multiStage_cute gemm_multiStage_cute.cu
# compute-sanitizer --tool memcheck res/gemm_multiStage_cute  > midRes/gemm_mS_cute_log.txt



# nvcc --generate-code arch=compute_86,code=sm_86 --expt-relaxed-constexpr -I/usr/local/cuda/include -I/home/yujixuan/kaiPro/cute-gemm-main/3rd/cutlass/include -L/usr/local/cuda/lib64 -l cuda -l cublas -o res/gemm_multiStage_cute gemm_multiStage_cute.cu

# res/gemm_multiStage_cute  > midRes/gemm_mS_cute_log.txt



# ncu -f -o midRes/gemm_mS_cute_prof_332 --set full --cache-control=all \
#  --clock-control=base res/gemm_multiStage_cute






################## new ############

nvcc --generate-code arch=compute_86,code=sm_86 --expt-relaxed-constexpr -I/usr/local/cuda/include -I/home/yujixuan/kaiPro/cute-gemm-main/3rd/cutlass/include -L/usr/local/cuda/lib64 -l cuda -l cublas -o res/gemm_multiStage_cute_1013 gemm_multiStage_cute_1013.cu

# res/gemm_multiStage_cute_1013  > midRes/gemm_mS_cute_log_1013.txt

ncu -f -o midRes/gemm_mS_cute_prof_1013 --set full --cache-control=all \
 --clock-control=base res/gemm_multiStage_cute_1013