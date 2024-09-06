
~~~bash
docker pull registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6
sudo docker run --gpus all --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN --name flash_decoding -v $PWD:/tyk --network=host -it paddlepaddle/paddle:2.6.1-gpu-cuda12.0-cudnn8.9-trt8.6  /bin/bash

~~~


- 如何划分任务给block
  - 这里的wave是啥意思 不是很懂
~~~C++
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
~~~