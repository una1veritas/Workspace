# エラトステネスのふるい
sievesize = 200
n = 43

prime = []
for i in range(sievesize):
    prime.append(True)
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
        last = (count, i)
        if count == n :
            break
print("見つかった最後の素数は", count, "番目で", i)