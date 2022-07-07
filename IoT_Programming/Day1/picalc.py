#ライプニッツの式による。
#π/4 を求めるものである．
#ここでは i 番目の pi4 とその直前の値 prev の平均の 4 倍（和の2倍）としてπを求める

import sys

error = float(sys.argv[1])

pi4 = 1
prev = 0
sgn = -1

i = 1
while 2*abs(pi4-prev) > error:
    prev = pi4
    pi4 += sgn/(2*i+1)
    sgn = -sgn
    i += 1

print("i = "+str(i), " pi ~= " + str( 2*(prev + pi4) ))
