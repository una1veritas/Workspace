pi = 2  
prev = 0
# 最後のループ後の値とその前回の値の和をとることにして 2 * 1/4 pi を計算する
sgn = -1

# ループ一回での変化量がこの値を下回ったら収束したとして終了させる
target_error = 1E-4

i = 1
deviation = 1  # 観察用、差がこの値を下回ったら印字出力
while True:
    prev = pi
    pi += sgn*2/(2*i+1)
    sgn = -sgn
    # PI の計算には直接関係のない、観察用の印字出力部
    if ( abs(pi - prev) < deviation ) :
        print(prev+pi, "(after " + str(i) + " iteration)")
        if deviation <= target_error :
            break
        deviation /= 10  # 次の目標はさらに 1/10
    # 印字出力部おわり
    i += 1

print("pi ~ " + str( prev + pi ))
