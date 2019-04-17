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

print("gcd for 141, 93 is " + str(gcd(141,93)))

# コマンドライン引数があれば最初の引数を整数と解釈し入力とする
n = 100
if len(sys.argv) > 1 :
    n = int(sys.argv[1])

print("Input: " + str(n))
print("Result: " + str(sieve(n)))
