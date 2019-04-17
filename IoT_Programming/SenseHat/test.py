# -*- coding: utf-8 -*-
import sys

def gcd(a, b):
    if a == 0 or b == 0 :
        return 0  # error
    while b != 0 :
        c = a % b
        a = b
        b = c
    return a

print(sys.argv)
if len(sys.argv) < 3:
    print("エラー: 引数が少なすぎです")
    exit(1) # エラー終了
    
x = int(sys.argv[1])
y = int(sys.argv[2])
print(gcd(x, y))

from datetime import datetime

def dt(form = 0):
    if form == 0:
        return datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    else:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


print(dt())
print(dt(form=1))
