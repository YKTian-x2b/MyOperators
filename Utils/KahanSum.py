def KahanSum(input):
    sum = 0.0   # 最终求和结果
    c = 0.0     # 运行时损失 减c实际上是加 sum和y的真值 与 sum+y的计算值 间的损失
    for i in range(len(input)):
        y = input[i] - c
        t = sum + y
        c = (t - sum) - y
        sum = t
    return sum
        
