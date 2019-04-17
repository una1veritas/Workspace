# ループの前後で差がこの値を下回ったら終了させる
target_error = 0.0000001

pi = 2  # あとでループの前後の値の和をとるので 1/2 pi を計算する
prev = 0
sgn = -1
i = 1
achieved_deviation = 0.1
while True:
    prev = pi
    pi += sgn*2/(2*i+1)
    sgn = -sgn
    # PI の計算には直接関係ない，収束した桁が増えたら表示，また目標の桁まで収束したら終了するための部分
    if ( abs(pi-prev) < achieved_deviation ) :
        print(prev+pi, i)
        if achieved_deviation <= target_error :
            break
        achieved_deviation /= 10
    # 表示部おわり
    i += 1

print("pi ~ " + str( prev + pi ))
