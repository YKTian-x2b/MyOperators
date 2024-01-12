
def DMAtransfer:
    for i in range(block):
        DMA_get(X[i])
        DMA_wait

        Comp:Y[i]=f(X[i])

        DMA_put(Y[i])
        DMA_wait



def doubleBuffering:
    inp_buff[2]
    out_buff[2]

    # prologue
    DMA_get(inp_buff[0],x[0], event_in[0])
    DMA_wait(event_in[0])
    DMA_get(inp_buff[1],x[1], event_in[1])
    # Comp:
    out_buff[0]=f(inp_buff[0]); 
    DMA_wait(event_in[1])

    # iteration
    for i in range (1, block-1):
        DMA_get(inp_buff[(i+1)%2], X[i+1], event_in[(i+1)%2])
        # Comp:
        out_buff[i%2]=f(inp_buff[i%2])
        
        DMA_put(out_buff[(i-1)%2], even_out[(i-1)%2])

        DMA_wait(event_in[(i+1)%2])
        DMA_wait(even_out[(i-1)%2])

    # epilogue
    # Comp:
    out_buff[(block-1)%2]=f(inp_buff[(block-1)%2])
    DMA_put(out_buff[(block-2)%2], even_out[(block-2)%2])
    DMA_wait(even_out[(block-2)%2])
    DMA_put(out_buff[(block-1)%2], even_out[(block-1)%2])
