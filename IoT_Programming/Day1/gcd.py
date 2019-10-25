# -*- coding: utf-8 -*-
import sys

# ユークリッドの互除法
def gcd(x, y):
    if x == 0 or y == 0 :
        return 0 # not defined
    while True :
        c = x % y
        if c == 0 :
            break
        x = y
        y = c
    return y


# コマンドライン引数の最初のふたつを入力とする

if len(sys.argv) >= 3 :
    a = int(sys.argv[1])
    b = int(sys.argv[2])
else:
    print(u"整数の引数を２つ受け取ります。")
    exit()

print("Input: {0}, {1}".format(a, b)) # 文字列 .format 関数を使う
print("Result: " + str(gcd(a, b)))    # 文字列どうしの結合演算を使う
