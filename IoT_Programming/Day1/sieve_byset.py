import sys

# エラトステネスのふるい
# 最小範囲は 100 まで，コマンドライン引数があれば最初の引数を範囲とする．

n = 1024
ith = 0

if len(sys.argv) > 1 :
    ith = max(ith, int(sys.argv[1]) )

while True :
    print("With array of size " + str(n) + ".")
    pset = set(range(2, n+1)) # range を set に変換

    for i in range(2, n+1) :
        for c in range(i*2, n+1, i):
            pset.discard(c)
    
    if ith == 0 :
        break
    if len(pset) >= ith :
        break
    n *= 2

print("the number of found primes: " + str(len(pset)))
parray = sorted(pset)
if ith :
    print(parray[ith-1])
else:
    print(parray)
