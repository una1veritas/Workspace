import sys

n = int(sys.argv[1])
print('going to find the {}th prime.'.format(n))

# True が 3,000 個ならんだ
# リストの作り方はいくつかある。
primes = [True]*3000
# primes = [ True for i in range(3000)]
#
# primes = []
# for i in range(3000):
#    primes.append(True)
#

nth = 0
for i in range(2,3000):
    if primes[i] :
        nth += 1
        print('{}th prime is {}.'.format(nth,i))
        if nth == n :
            break
        c = 2
        while c * i < 3000:
            primes[i * c] = False
            c += 1

