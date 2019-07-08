# -*- coding: utf-8 -*-

pi = 1
prev = 0
sgn = -1

#ライプニッツの式は π/4 を求める．
#ここでは i 番目の pi とその直前の値 prev の平均の 4 倍（和の2倍）としてπを求める
i = 1
while i < 10:
    prev = pi
    pi += sgn/(2*i+1)
    sgn = -sgn
    i += 1

print("pi ~= " + str( 2*(prev + pi) ))
