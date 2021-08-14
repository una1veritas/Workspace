import sys

# エラトステネスのふるい
# 最小範囲は 100 まで，コマンドライン引数があれば最初の引数を範囲とする．

n = 100
if len(sys.argv) > 1 :
    n = max(n, int(sys.argv[1]) )

parray = [True if i > 1 else False for i in range(n+1)] 
# 上行は，以下と同等の内包記法．
# pnums = []
# for i in range(n):
#    if i < 1 :
#       pnums.append(True)
#    else:
#       pnums.append(True)

for i in range(2, n+1) :
    for c in range(i*2, n+1, i):
        parray[c] = False

print("Input: " + str(n))
for i in range(n+1):
    if parray[i] :
        print(i, end=", ")
