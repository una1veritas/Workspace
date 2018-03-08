import sys

print(sys.argv)

n = int(sys.argv[1])

f = 1
f2 = 1
for i in range(1, n):
    print(str(i)+': '+str(f))
    fnext = f + f2
    f = f2
    f2 = fnext
    
    