import sys

# エラトステネスのふるい
# 最小範囲は 100 まで，コマンドライン引数があれば最初の引数を範囲とする．

n = 100
ith = 0
if len(sys.argv) > 1 :
    n = max(n, int(sys.argv[1]) )
if len(sys.argv) > 2 :
    ith = max(ith, int(sys.argv[2]) )

pset = set(range(2, n+1)) # range を set に変換

for i in range(2, n+1) :
    for c in range(i*2, n+1, i):
        pset.discard(c)

print("Input: " + str(n))
parray = sorted(pset)
if ith :
    print(parray[ith+1])
else:
    print(parray)