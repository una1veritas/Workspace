# -*- coding: utf-8 -*-
import sys

# エラトステネスのふるい
def sieve(n):
    primes = set(range(2, n+1) ) 
    for i in range(2, int((n+1)/2) ) :
        mul = 2
        while (mul*i <= n+1) :
            if mul*i in primes :
                primes.remove(mul*i) 
            mul = mul + 1
    return primes

# コマンドライン引数があれば最初の引数を整数と解釈し入力とする
n = 100
if len(sys.argv) > 1 :
    n = int(sys.argv[1])

print("Search for prime numbers up to " + str(n) + ".")
print("Result: " + str(sieve(n)))
