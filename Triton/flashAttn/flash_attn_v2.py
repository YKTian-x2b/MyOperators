import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import triton
import triton.language as tl
import torch 
import numpy as np

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, }, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, }, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, }, num_stages=5, num_warps=2),
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
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_seq = tl.program_id(axis=0)
    pid_bz_nh = tl.program_id(axis=1)
    qkv_offset = pid_bz_nh * SEQ_LEN * HEAD_DIM

    Q_blk_ptr = tl.make_block_ptr(base=Q + qkv_offset,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(pid_seq*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                               order=(1, 0),)
    # 这块列主序没搞懂 待思考
    K_blk_ptr = tl.make_block_ptr(base=K + qkv_offset,
                                    shape=(HEAD_DIM, SEQ_LEN),
                                    strides=(1, HEAD_DIM),
                                    offsets=(0, 0),
                                    block_shape=(HEAD_DIM, BLOCK_SIZE_N),
                                    order=(0, 1),)
    V_blk_ptr = tl.make_block_ptr(base=V + qkv_offset,
                                      shape=(SEQ_LEN, HEAD_DIM),
                                      strides=(HEAD_DIM, 1),
                                      offsets=(0, 0),
                                      block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                                      order=(1, 0),)
    
    
    S_max_prev = tl.full((BLOCK_SIZE_M, 1), 0. - float("inf"), dtype=tl.float32)
    S_sum_prev = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    O_eles_prev = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    for i in range(SEQ_LEN // BLOCK_SIZE_N):
        Q_eles = tl.load(Q_blk_ptr)
        K_eles = tl.load(K_blk_ptr)
        # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        S_eles = tl.dot(Q_eles, K_eles)
        S_eles = S_eles * scale_dh
        
        #################################### softmax ##########################################
        # m(i): [BLOCK_SIZE_M, 1]
        S_max_curr = tl.max(S_eles, axis=1, keep_dims=True)
        # m, 希望它是[BLOCK_SIZE_M, 1]的
        S_max_curr = tl.maximum(S_max_curr, S_max_prev)
        
        # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        S_eles = tl.exp(S_eles - S_max_curr)
        
        # l(i): [BLOCK_SIZE_M, 1]
        S_sum_curr = tl.sum(S_eles, axis=1, keep_dims=True)
        # l
        S_sum_prev = S_sum_curr + S_sum_prev * tl.exp(S_max_prev - S_max_curr)
        ########################################################################################
        
        V_eles = tl.load(V_blk_ptr)
        # O: [BLOCK_SIZE_M, HEAD_DIM]
        O_eles_prev = O_eles_prev * tl.exp(S_max_prev - S_max_curr) + tl.dot(S_eles.to(tl.float16), V_eles)
        
        S_max_prev = S_max_curr
        K_blk_ptr = tl.advance(K_blk_ptr, (0, BLOCK_SIZE_N))
        V_blk_ptr = tl.advance(V_blk_ptr, (BLOCK_SIZE_N, 0))
        
    O_blk_ptr = tl.make_block_ptr(base=O + qkv_offset,
                                  shape=(SEQ_LEN, HEAD_DIM),
                                  strides=(HEAD_DIM, 1),
                                  offsets=(pid_seq*BLOCK_SIZE_M, 0),
                                  block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                                  order=(1, 0),)
    O_eles_prev = (O_eles_prev / S_sum_prev).to(tl.float16)
    tl.store(O_blk_ptr, O_eles_prev)


def call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM):
    BLOCK_SIZE_N_ = 128
    assert SEQ_LEN % BLOCK_SIZE_N_ == 0
  
    O = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    scale_dh = 1.0 / (HEAD_DIM**0.5)
    grid = lambda META: (triton.cdiv(SEQ_LEN, META['BLOCK_SIZE_M']), BSZ*NUM_HEAD, 1)
    flash_attn_fwd[grid](
        Q, K, V, O, scale_dh, 
        NUM_HEAD, SEQ_LEN, HEAD_DIM,
        BLOCK_SIZE_N = BLOCK_SIZE_N_,
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
    SEQ_LEN = 128 * 8
    HEAD_DIM = 64
    
    Q = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    K = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    V = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    O_triton = call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    O_ref = call_ref_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    diff_ = (O_triton - O_ref).cpu()
    print("O_triton", O_triton[0, 1, -1, :])
    print("O_ref", O_ref[0, 1, -1, :])
    print("diff_:", diff_[0, 1, -1, :])
    
    max_abs_diff = torch.max(torch.abs(O_triton - O_ref)).item()
    print("O max abs diff: ", max_abs_diff)
    # print("O max rel diff:", max_abs_diff / (torch.mean(torch.abs(O_ref)).item()))
    print("O allclose: ", torch.allclose(O_triton, O_ref, rtol=0, atol=1e-02))