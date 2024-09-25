
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
