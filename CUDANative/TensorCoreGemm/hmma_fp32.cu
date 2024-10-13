
// mma.sync.aligned.m16n8k8.row.col.dtype.f16.f16.ctype  d, a, b, c;
// D(16x8) = A(16x8) * B(8x8) + C(16x8) warp中每个线程处理16*8/32=4个a，2个b。
// 因为a和b元素占半个寄存器，所以用2个a寄存器，1个b寄存器。

template <>
inline __device__ void hmma_fp32(float4& c, uint2 a, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
        "    {%0, %1, %2, %3}, \n"
        "    {%4, %5}, \n"
        "    {%6}, \n"
        "    {%0, %1, %2, %3}; \n"
        : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
        : "r"(a.x), "r"(a.y), "r"(b));
}



// ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];
// .shape  = {.m8n8};
// .num    = {.x1, .x2, .x4};
// .ss     = {.shared{::cta}};
// .type   = {.b16};

__host__ __device__ static void
copy(uint128_t const& smem_src,
    uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
{
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
}



CUTE_DEVICE
uint32_t
cast_smem_ptr_to_uint(void const* const ptr)
{
  uint32_t smem_ptr;
  asm(
  "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
    : "=r"(smem_ptr) : "l"(ptr));

  return smem_ptr;
}