# エラトステネスのふるい
import math, sys

n = 43
if len(sys.argv) > 1 :
    n = int(sys.argv[1])

sievesize = 200 if n < 50 else int(n * (math.ceil(1.2 * math.log(n, math.e))))

prime = [True] * sievesize
prime[0] = False   # primes[0] は使わない
prime[1] = False   # 1 は素数でない

for i in range(2, sievesize) :
    if prime[i] :
        for c in range(i*2, sievesize, i):  # i の整数倍について
            prime[c] = False

count = 0
for i in range(sievesize):
    if prime[i] :
        count += 1
        if count == n :
            print("{} 番目の素数は {}".format(n, i))
            break
else:
    print("sievesize = {} で {} 番目を見つける計算失敗".format(sievesize, n))