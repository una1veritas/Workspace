# -*- coding: utf-8 -*-
import sys

# ユークリッドの互除法
def gcd(x, y):
    if x == 0 or y == 0 :
        return 0 # not defined
    while True :
        c = x % y
        x = y
        y = c
        if y == 0 :
            break
    return x

if len(sys.argv) < 3 :
    print("Give two numbers.")
    exit(1)
a = int(sys.argv[1])
b = int(sys.argv[2])
print("gcd for " + str(a) + " and " + str(b) + " is: ")
print(gcd(a, b))
