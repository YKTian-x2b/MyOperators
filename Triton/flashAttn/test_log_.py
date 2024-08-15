import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import triton
import triton.language as tl
import torch 
import numpy as np

# triton.Config({'BLOCK_SIZE_M': 64, }, num_stages=4, num_warps=4),
# triton.Config({'BLOCK_SIZE_M': 32, }, num_stages=5, num_warps=2),
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, }, num_stages=3, num_warps=8),
    ],
    key=['SEQ_LEN', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_fwd_v2(
    Q, K, V, O, scale_dh: tl.float16, midRes, 
    NUM_HEAD, 
    SEQ_LEN: tl.constexpr, 
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_seq = tl.program_id(axis=0)
    pid_bz_nh = tl.program_id(axis=1)
    qkv_offset = pid_bz_nh * SEQ_LEN * HEAD_DIM
    # midRes_offset = pid_bz_nh * SEQ_LEN * SEQ_LEN
    midRes_offset = pid_bz_nh * SEQ_LEN

    Q_blk_ptr = tl.make_block_ptr(base=Q + qkv_offset,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(pid_seq*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                               order=(1, 0),)
    
    S_max_prev = tl.full((BLOCK_SIZE_M, 1), 0. - float("inf"), dtype=tl.float32)
    S_sum_prev = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)
    O_eles_prev = tl.zeros((BLOCK_SIZE_M, HEAD_DIM), dtype=tl.float32)
    for i in range(SEQ_LEN // BLOCK_SIZE_N):
        N_offset = i * BLOCK_SIZE_N
        # 这块列主序没搞懂 待思考
        K_blk_ptr = tl.make_block_ptr(base=K + qkv_offset,
                                      shape=(HEAD_DIM, SEQ_LEN),
                                      strides=(1, HEAD_DIM),
                                      offsets=(0, N_offset),
                                      block_shape=(HEAD_DIM, BLOCK_SIZE_N),
                                      order=(0, 1),)
        
        Q_eles = tl.load(Q_blk_ptr).to(tl.float32)
        K_eles = tl.load(K_blk_ptr).to(tl.float32)
        # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        S_eles = tl.dot(Q_eles, K_eles)

        S_eles = S_eles * scale_dh
        
        #################################### softmax ##########################################
        # m(i): [BLOCK_SIZE_M, 1]
        S_max = tl.max(S_eles, axis=1, keep_dims=True)
        # m: [BLOCK_SIZE_M, 1]
        S_max = tl.maximum(S_max, S_max_prev)
        
        # [BLOCK_SIZE_M, BLOCK_SIZE_N]
        S_eles = tl.exp(S_eles - S_max)
        
        # l(i): [BLOCK_SIZE_M, 1]
        S_sum = tl.sum(S_eles, axis=1, keep_dims=True)
        # l
        S_sum = S_sum + S_sum_prev * tl.exp(S_max_prev - S_max)
        
        
        #----------############ midRes #############
        midRes_blk_ptr = tl.make_block_ptr(base=midRes + midRes_offset,
                               shape=(SEQ_LEN, 1),
                               strides=(1, 1),
                               offsets=(pid_seq*BLOCK_SIZE_M, 0),
                               block_shape=(BLOCK_SIZE_M, 1),
                               order=(1, 0),)
        tl.store(midRes_blk_ptr, S_sum.to(tl.float16))
        #----------################################
        
        
        
        # p(i): [BLOCK_SIZE_M, BLOCK_SIZE_N]
        P_eles = S_eles / S_sum
        ########################################################################################
        
        
        V_blk_ptr = tl.make_block_ptr(base=V + qkv_offset,
                                      shape=(SEQ_LEN, HEAD_DIM),
                                      strides=(HEAD_DIM, 1),
                                      offsets=(N_offset, 0),
                                      block_shape=(BLOCK_SIZE_N, HEAD_DIM),
                                      order=(1, 0),)
        V_eles = tl.load(V_blk_ptr).to(tl.float32)
        # O: [BLOCK_SIZE_M, HEAD_DIM]
        O_eles_prev = O_eles_prev * S_sum_prev / S_sum + tl.dot(P_eles, V_eles)
        
        S_max_prev = S_max
        S_sum_prev = S_sum
        
    O_blk_ptr = tl.make_block_ptr(base=O + qkv_offset,
                                  shape=(SEQ_LEN, HEAD_DIM),
                                  strides=(HEAD_DIM, 1),
                                  offsets=(pid_seq*BLOCK_SIZE_M, 0),
                                  block_shape=(BLOCK_SIZE_M, HEAD_DIM),
                                  order=(1, 0),)
    tl.store(O_blk_ptr, O_eles_prev.to(tl.float16))


def call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM):
    BLOCK_SIZE_N_ = 64
    assert SEQ_LEN % BLOCK_SIZE_N_ == 0
  
    O = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    scale_dh = 1.0 / (HEAD_DIM**0.5)
    
    # 为了对精度
    # midRes = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, SEQ_LEN)
    midRes = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, 1)
    
    grid = lambda META: (triton.cdiv(SEQ_LEN, META['BLOCK_SIZE_M']), BSZ*NUM_HEAD, 1)
    flash_attn_fwd_v2[grid](
        Q, K, V, O, scale_dh, midRes, 
        NUM_HEAD, SEQ_LEN, HEAD_DIM,
        BLOCK_SIZE_N = BLOCK_SIZE_N_,
    )
    return O, midRes


def call_ref_flash_attn_fwd(q, k, v, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM):
    sm_scale = 1.0 / (HEAD_DIM**0.5)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p_max = torch.max(p, dim=-1, keepdim=True)[0]
    p = p - p_max
    
    p_exp = p.exp()
    
    partition = p_exp.sum(dim=-1, keepdim=True)
    midRes = partition
    
    p = (p_exp / partition).half()
    
    

    ref_out = torch.matmul(p, v)
    return ref_out, midRes


def print_diff(triton_res, ref_res, msg:str, log_path):
    diff_ = (triton_res - ref_res).cpu()
    max_abs_diff = torch.max(torch.abs(diff_)).item()
    with open(log_path, "a") as wf:
        content = ''
        content += f"{msg}_triton:\n {triton_res[0, 1, -1, :]}\n"
        content += f"{msg}_ref:\n {ref_res[0, 1, -1, :]}\n"
        content += f"{msg}_diff:\n {diff_[0, 1, -1, :]}\n"
        
        content += f"{msg} max abs diff: {max_abs_diff}\n"
        content += f"\nall diff: \n{diff_}\n"
        content += f"{msg} allclose: {torch.allclose(ref_res, triton_res, rtol=0, atol=1e-02)}\n\n\n"
        wf.write(content)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    torch.set_default_device('cuda')
    
    BSZ = 1
    NUM_HEAD = 2
    SEQ_LEN = 128 * 2
    HEAD_DIM = 64
    
    Q = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    K = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    V = torch.randn(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    O_triton, midRes_triton = call_triton_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    O_ref, midRes_ref = call_ref_flash_attn_fwd(Q, K, V, BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    
    
    log_path = "/tyk/kaiPro/MyOperators/Triton/flashAttn/log.txt"
    with open(log_path, 'a+', encoding='utf-8') as wf:
        wf.truncate(0)

    print_diff(midRes_triton, midRes_ref, "midRes", log_path)
    print_diff(O_triton, O_ref, "O", log_path)
    print("done!")