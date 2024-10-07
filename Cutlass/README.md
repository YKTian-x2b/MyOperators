
// ArgType          ArgName          DataType               Shape                           Layout
//
// input            act              fp16/bf16              [m, k]                          RowMajor
// input            act_scale        fp16/bf16              [1, k]                          RowMajor
// input            weight           int4b/int8b            [k, n]                          ColumnMajor or ColumnMajorInterleaved
// input            scales           fp16/bf16              [k / GroupSize, n] or [1, n]    RowMajor
// input            zeros            fp16/bf16              [k / GroupSize, n] or [1, n]    RowMajor
// input            bias             fp16/bf16              [1, n]                          RowMajor
// output           out              fp16/bf16              [m, n]                          RowMajor


interleave the weight tile from (CtaShapeN, CtaShapeK) to (CtaShapeN/ColumnsInterleaved, CtaShapeK * ColumnsInterleaved)


- ThreadblockSwizzle
~~~C++
  // params.swizzle_log_tile = ThreadblockSwizzle().get_log_tile(grid_tiled_shape)
  __forceinline__ __device__ __host__ static int get_log_tile(GemmCoord tiled_shape) {
    auto n = tiled_shape.n();
    if (n >= 6)
      return 3;
    else if (n >= 3)
      return 2;
    else if (n >= 2)
      return 1;
    else
      return 0;
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  __forceinline__ __device__ static GemmCoord get_tile_offset(int log_tile) {
    return GemmCoord{(blockIdx.x >> log_tile),  //
                     (blockIdx.y << log_tile) + ((blockIdx.x) & ((1 << (log_tile)) - 1)),
                     blockIdx.z};
  }

  // GemmCoord threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);
~~~