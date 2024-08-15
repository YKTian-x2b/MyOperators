import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import triton
import triton.language as tl
import torch 
import numpy as np


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, }, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, }, num_stages=5, num_warps=2),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_fwd(
    Q, K, V, O, scale_dh: tl.float32, 
    NUM_HEAD, 
    SEQ_LEN: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_seq = tl.program_id(axis=0)
    pid_bz_nh = tl.program_id(axis=1)
    pid_bz = pid_bz_nh // NUM_HEAD
    pid_nh = pid_bz_nh % NUM_HEAD
    qkv_offset = pid_bz_nh * SEQ_LEN * HEAD_DIM

    Q_blk_ptr = tl.make_block_ptr(base=Q + qkv_offset,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(pid_seq*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                               order=(1, 0),)

    K_blk_ptr = tl.make_block_ptr(base=K + qkv_offset,
                               shape=(HEAD_DIM, SEQ_LEN),
                               strides=(1, HEAD_DIM),
                               offsets=(0, 0),
                               block_shape=(HEAD_DIM, SEQ_LEN),
                               order=(0, 1),)
    
    Q_eles = tl.load(Q_blk_ptr).to(tl.float32)
    K_eles = tl.load(K_blk_ptr).to(tl.float32)
    # [BLOCK_SIZE_M, SEQ_LEN]
    S_eles = tl.dot(Q_eles, K_eles)
    
    S_eles = S_eles * scale_dh
    # [BLOCK_SIZE_M, 1]
    S_max = tl.max(S_eles, axis=1, keep_dims=True)
    # [BLOCK_SIZE_M, SEQ_LEN]
    S_eles = tl.exp(S_eles - S_max)
    # [BLOCK_SIZE_M, 1]
    S_sum = tl.sum(S_eles, axis=1, keep_dims=True)
    P_eles = S_eles / S_sum

    V_blk_ptr = tl.make_block_ptr(base=V + qkv_offset,
                                  shape=(SEQ_LEN, HEAD_DIM),
                                  strides=(HEAD_DIM, 1),
                                  offsets=(0, 0),
                                  block_shape=(SEQ_LEN, HEAD_DIM),
                                  order=(1, 0),)
    V_eles = tl.load(V_blk_ptr).to(tl.float32)
    O_eles = tl.dot(P_eles, V_eles)
    
    O_blk_ptr = tl.make_block_ptr(base=O + qkv_offset,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(pid_seq*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                               order=(1, 0),)
    tl.store(O_blk_ptr, O_eles.to(tl.float16))
    


def call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM):
    O = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    scale_dh = 1.0 / (HEAD_DIM**0.5)
    grid = lambda META: (triton.cdiv(SEQ_LEN, META['BLOCK_SIZE_M']), BSZ*NUM_HEAD, 1)
    flash_attn_fwd[grid](
        Q, K, V, O, scale_dh, 
        NUM_HEAD, SEQ_LEN, HEAD_DIM,
    )
    return O


def call_ref_flash_attn_fwd(q, k, v, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM):
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p_max = torch.max(p, dim=-1, keepdim=True)[0]
    p = p - p_max
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    return ref_out


if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    torch.set_default_device('cuda')
    
    BSZ = 1
    NUM_HEAD = 2
    SEQ_LEN = 256
    HEAD_DIM = 64
    
    Q = torch.rand(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    K = torch.rand(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    V = torch.rand(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    O_triton = call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    O_ref = call_ref_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    diff_ = (O_triton - O_ref).cpu()
    print("O_triton", O_triton[0, 0, 0, :])
    print("O_ref", O_ref[0, 0, 0, :])
    print("diff_:", diff_[0, 0, 0, :])
    
    max_abs_diff = torch.max(torch.abs(O_triton - O_ref)).item()
    print("O max abs diff: ", max_abs_diff)
    print("O max rel diff:", max_abs_diff / (torch.mean(O_ref).item()))
    print("O allclose: ", torch.allclose(O_triton, O_ref, rtol=1e-03, atol=1e-02))