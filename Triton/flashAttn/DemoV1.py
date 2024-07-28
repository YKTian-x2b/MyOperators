import triton
import triton.language as tl
import torch 

torch.set_default_dtype(torch.float16)
torch.set_default_device('cuda')


# 没搞清楚层次结构啊 小田
@triton.jit
def flash_attn_fwd(
    Q, K, V, O,
    BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    Q_blk_ptr = tl.make_block_ptr(base=Q,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(pid*BLOCK_SIZE, 0),
                               block_shape=(BLOCK_SIZE, HEAD_DIM),
                               order=(1, 0),)

    K_blk_ptr = tl.make_block_ptr(base=K,
                               shape=(SEQ_LEN, HEAD_DIM),
                               strides=(HEAD_DIM, 1),
                               offsets=(0, 0),
                               block_shape=(SEQ_LEN, HEAD_DIM),
                               order=(1, 0),)
    
    Q_eles = tl.load(Q_blk_ptr)
    K_eles = tl.load(K_blk_ptr)
    # [BLOCK_SIZE, HEAD_DIM]
    S_eles = tl.dot(Q_eles, K_eles)
    S_eles = S_eles * tl.rsqrt(HEAD_DIM)
    # [BLOCK_SIZE, 1]
    S_max = tl.max(S_eles, axis=1, keep_dims=True)
    # [BLOCK_SIZE, HEAD_DIM]
    S_eles = tl.exp(S_eles - S_max)
    # [BLOCK_SIZE, 1]
    S_sum = tl.sum(S_eles, axis=1, keep_dims=True)
    P_eles = S_eles / S_sum

    V_blk_ptr = tl.make_block_ptr(base=V,
                                  shape=(SEQ_LEN, HEAD_DIM),
                                  strides=(HEAD_DIM, 1),
                                  offsets=(0, 0),
                                  block_shape=(SEQ_LEN, HEAD_DIM),
                                  order=(1, 0),)
    V_eles = tl.load(V_blk_ptr)
    O_eles = tl.dot(P_eles, V_eles)
    # tl.store(O, O_eles)

    



if __name__ == "__main__":
    BSZ = 2
    SEQ_LEN = 2 << 12 # 4K => 32K 
    NUM_HEAD = 32
    HEAD_DIM = 128

    M = BSZ*NUM_HEAD 
    
    Q = torch.rand(BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM)
    K = torch.rand(BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM)
    V = torch.rand(BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM)
    O = torch.empty(BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM)

    Q = tl.trans(Q, (0, 2, 1, 3))
    K = tl.trans(K, (0, 2, 1, 3))
    V = tl.trans(V, (0, 2, 1, 3))
    O = tl.trans(O, (0, 2, 1, 3))
    print("Q.shape", Q.shape)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE']))
    flash_attn_fwd[grid](
        Q, K, V, O,
        BSZ, SEQ_LEN, NUM_HEAD, HEAD_DIM,
    )

