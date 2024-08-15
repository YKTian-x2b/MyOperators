#pragma once

#include <string>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"



template <typename T, int Dh, bool DO_CROSS_ATTENTION>
inline void multi_block_grid_setup(dim3& grid, Multihead_attention_params<T, DO_CROSS_ATTENTION> const& params,
    int blocks_per_sm, int block_size, int tlength)
{
    if (!params.multi_block_mode)
    {
        return;
    }

    int balanced_seq_len_tile
        = mmha::divUp(params.multi_processor_count * blocks_per_sm, params.batch_size * params.num_heads);

    int const threads_per_value = mmha::threads_per_value<T>(mmha::dh_max(Dh));
    // Make sure that each block at least processes one loop of kv (unroll size is default at 8).
    int const seq_len_per_kv_loop = mmha::divUp(block_size, threads_per_value) * 8;
    int max_seq_len_tile = params.max_seq_len_tile;

    max_seq_len_tile = std::min(mmha::divUp(tlength + 1, seq_len_per_kv_loop), max_seq_len_tile);
    

    params.seq_len_tile = std::clamp(balanced_seq_len_tile, params.min_seq_len_tile, max_seq_len_tile);

    TLLM_CHECK_WITH_INFO(
        params.seq_len_tile <= block_size, "The number of blocks per sequence may not exceed the thread block size.");

    // We should consider the new timestep.
    params.timesteps_per_block
        = mmha::divUp(std::min(tlength, params.cyclic_attention_window_size) + 1, params.seq_len_tile);

    params.multi_block_mode = (params.seq_len_tile > 1);



    grid.z = params.seq_len_tile;
}



inline gpuError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return gpuSuccess;
}