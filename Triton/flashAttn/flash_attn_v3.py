import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import triton
import triton.language as tl
import torch 
import numpy as np

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, }, num_stages=3, num_warps=8),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_fwd(
    Q, K, V, O_split, O, 
    lock_ptr, scale_dh: tl.float32, 
    NUM_HEAD, 
    SEQ_LEN: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLK_NUM_IN_SEQ: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)   # key value 维度的切分
    pid_m = tl.program_id(axis=1)   # query 维度的切分
    pid_bz_nh = tl.program_id(axis=2)
    qkv_offset = pid_bz_nh * SEQ_LEN * HEAD_DIM
    o_split_offset = pid_bz_nh * BLK_NUM_IN_SEQ * SEQ_LEN * HEAD_DIM
    num_seq_per_blk = SEQ_LEN // (BLOCK_SIZE_N * BLK_NUM_IN_SEQ)

    Q_blk_ptr = tl.make_block_ptr(base=Q + qkv_offset,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(pid_m*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                               order=(1, 0),)
    # 这块列主序没搞懂 待思考
    K_blk_ptr = tl.make_block_ptr(base=K + qkv_offset,
                                    shape=(HEAD_DIM, SEQ_LEN),
                                    strides=(1, HEAD_DIM),
                                    offsets=(0, pid_n * num_seq_per_blk * BLOCK_SIZE_N),
                                    block_shape=(HEAD_DIM, BLOCK_SIZE_N),
                                    order=(0, 1),)
    V_blk_ptr = tl.make_block_ptr(base=V + qkv_offset,
                                      shape=(SEQ_LEN, HEAD_DIM),
                                      strides=(HEAD_DIM, 1),
                                      offsets=(pid_n * num_seq_per_blk * BLOCK_SIZE_N, 0),
                                      block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                                      order=(1, 0),)
    
    S_max_prev = tl.full((BLOCK_SIZE_M, 1), 0. - float("inf"), dtype=tl.float32)
    S_sum_prev = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    O_eles_prev = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    for i in range(num_seq_per_blk):
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
        
    O_split_blk_ptr = tl.make_block_ptr(base=O_split + o_split_offset,
                                        shape=(SEQ_LEN, BLK_NUM_IN_SEQ, HEAD_DIM),
                                        strides=(BLK_NUM_IN_SEQ*HEAD_DIM, HEAD_DIM, 1),
                                        offsets=(pid_m*BLOCK_SIZE_M, pid_n, 0),
                                        block_shape=(BLOCK_SIZE_M, 1, HEAD_DIM),
                                        order=(2, 1, 0),)
    # 现在的这个结果呢是1个block的，需要4个block的，拼接成一个正确的
    O_eles_prev = (O_eles_prev / S_sum_prev).to(tl.float16)
    O_eles_prev = tl.reshape(O_eles_prev, (BLOCK_SIZE_M, 1, HEAD_DIM))
    tl.store(O_split_blk_ptr, O_eles_prev)
    
    lock_ptr += pid_bz_nh
    prev_lock_val = tl.atomic_add(lock_ptr, 1, sem="acq_rel", scope="gpu")

    if(prev_lock_val == BLK_NUM_IN_SEQ-1):
        O_eles_res = tl.zeros((BLOCK_SIZE_M, 1, HEAD_DIM), dtype=tl.float32)
        
        O_split_blk_ptr = tl.make_block_ptr(base=O_split + qkv_offset,
                                            shape=(SEQ_LEN, BLK_NUM_IN_SEQ, HEAD_DIM),
                                            strides=(BLK_NUM_IN_SEQ*HEAD_DIM, HEAD_DIM, 1),
                                            offsets=(pid_m*BLOCK_SIZE_M, 0, 0),
                                            block_shape=(BLOCK_SIZE_M, 1, HEAD_DIM),
                                            order=(2, 1, 0),)
        # TODO:很显然 循环里不能只是加法操作
        for i in range(BLK_NUM_IN_SEQ):
            O_eles = tl.load(O_split_blk_ptr)
            O_eles_res = O_eles_res + O_eles
            O_split_blk_ptr = tl.advance(O_split_blk_ptr, (0, 1, 0))
        O_blk_ptr = tl.make_block_ptr(base=O + qkv_offset,
                                  shape=(SEQ_LEN, HEAD_DIM),
                                  strides=(HEAD_DIM, 1),
                                  offsets=(pid_m*BLOCK_SIZE_M, 0),
                                  block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                                  order=(1, 0),)
        O_eles_res = tl.reshape(O_eles_res, (BLOCK_SIZE_M, HEAD_DIM))
        tl.store(O_blk_ptr, O_eles_res.to(tl.float16))



def call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM, flash_decoding=False):
    O = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    scale_dh = 1.0 / (HEAD_DIM**0.5)
    
    BLK_NUM_IN_SEQ = 1
    if flash_decoding:
        # 实际上这个SEQs_PER_SPLIT 应该是算出来的 
        # num_block = sm_count * max_thread_per_multiprocessor / BlockSize 类似这种
        # num_block = num_block / (bsz * num_head * seq_len/block_size_m)
        BLK_NUM_IN_SEQ = 4
    lock = torch.zeros(BSZ, NUM_HEAD, dtype=torch.int32)
    O_split = torch.zeros(BSZ, NUM_HEAD, SEQ_LEN, BLK_NUM_IN_SEQ, HEAD_DIM)
    
    grid = lambda META: (BLK_NUM_IN_SEQ, triton.cdiv(SEQ_LEN, META['BLOCK_SIZE_M']), BSZ*NUM_HEAD)
    flash_attn_fwd[grid](
        Q, K, V, O_split, O, lock, scale_dh, 
        NUM_HEAD, 
        SEQ_LEN=SEQ_LEN, 
        HEAD_DIM=HEAD_DIM, 
        BLK_NUM_IN_SEQ=BLK_NUM_IN_SEQ,
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
    
    assert SEQ_LEN % 128 == 0   #
        
    Q = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    K = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    V = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    O_triton = call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM, flash_decoding=True)
    O_ref = call_ref_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    diff_ = (O_triton - O_ref).cpu()
    print("O_triton", O_triton[0, 1, -1, :])
    print("O_ref", O_ref[0, 1, -1, :])
    print("diff_:", diff_[0, 1, -1, :])
    
    max_abs_diff = torch.max(torch.abs(O_triton - O_ref)).item()
    print("O max abs diff: ", max_abs_diff)
    # print("O max rel diff:", max_abs_diff / (torch.mean(torch.abs(O_ref)).item()))
    print("O allclose: ", torch.allclose(O_triton, O_ref, rtol=0, atol=1e-02))