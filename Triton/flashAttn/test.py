import triton
import triton.language as tl
import torch 




if __name__ == "__main__":
    BSZ = 2
    SEQ_LEN = 2 << 10 # 4K => 32K 
    NUM_HEAD = 32
    HEAD_DIM = 128
    
    Q = torch.rand(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    K = torch.rand(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    V = torch.rand(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)
    O = torch.empty(BSZ, NUM_HEAD, SEQ_LEN, HEAD_DIM)

    all = BSZ * NUM_HEAD * SEQ_LEN * HEAD_DIM
    print(Q.stride(0))
    all = all // BSZ
    print(all)
    print(Q.stride(1))
    all = all // NUM_HEAD
    print(all)
    print(Q.stride(2))
    all = all // SEQ_LEN
    print(all)
    print(Q.stride(3))
    all = all // HEAD_DIM
    print(all)
 

