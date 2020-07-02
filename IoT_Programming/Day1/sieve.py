import sys

# エラトステネスのふるい
def sieve(n):
    primenums = set(range(2, n+1))
    for i in range(2, n+1) :
        p = i*2
        while p < n+1 :
            primenums.discard(p)
            p += i
    return primenums


# コマンドライン引数があれば最初の引数を入力とする
n = 100
if len(sys.argv) > 1 :
    n = int(sys.argv[1])

print("Input: " + str(n))
print("Result: " + str(sieve(n)))
