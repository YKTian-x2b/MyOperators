template<> class fragment<matrix_a, 16, 16, 16, __half, row_major> : public __frag_base<__half, 16> {};
template<> class fragment<matrix_a, 16, 16, 16, __half, col_major> : public __frag_base<__half, 16> {};
template<> class fragment<matrix_b, 16, 16, 16, __half, row_major> : public __frag_base<__half, 16> {};
template<> class fragment<matrix_b, 16, 16, 16, __half, col_major> : public __frag_base<__half, 16> {};
template<> class fragment<accumulator, 16, 16, 16, __half> : public __frag_base<__half, 8> {};
template<> class fragment<accumulator, 16, 16, 16, float> : public __frag_base<float, 8> {};


__CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, __half>& a, const __half* p, unsigned ldm, layout_t layout) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, float>& a, const float* p, unsigned ldm, layout_t layout) __DEF_IF_HOST


__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(__half *p, const fragment<accumulator, 16, 16, 16, __half>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 16, float>& a, unsigned ldm, layout_t layout) __DEF_IF_HOST
                 

__CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, col_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, row_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST
__CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator,16, 16, 16, __half>& d, const fragment<matrix_a, 16, 16, 16, __half, col_major>& a, const fragment<matrix_b,16, 16, 16, __half, row_major>& b, const fragment<accumulator,16, 16, 16, float>& c) __DEF_IF_HOST


void fill_fragment(__frag_base<FragEleType, size, packed_size> &f, 
    const typename helper_traits<FragEleType>::fill_argument_type &in);