template <typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 1, T> constexpr next_power_of_two(T v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    return ++v;
}

template <typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 2, T> constexpr next_power_of_two(T v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    return ++v;
}

template <typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 4, T> constexpr next_power_of_two(T v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

template <typename T>
__device__ __host__ std::enable_if_t<sizeof(T) == 8, T> constexpr next_power_of_two(T v)
{
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return ++v;
}