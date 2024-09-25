


- 如何按硬件能力划分任务给block
~~~C++
inline getBlocksPerSM(){
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(int numBlocks, const void func, int blockSize, size_t dynamicSMemSize)
    PADDLE_ENFORCE_GPU_SUCCESS(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&available_blocks,                                   
        mmha::masked_multihead_attention_kernel<>,                                                 
        DYNAMIC_THDS_PER_BLOCK, dynamic_smem_sz));
}

inline unsigned GetNumBlocks() {
  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);

  unsigned balanced_seq_len_tile =
        div_up(sm_count * getBlocksPerSM(),
              params.batch_size * params.num_heads);
  

  return balanced_seq_len_tile;
}
~~~