#python3
import sys

n = 15
if len(sys.argv) >= 2 :
    n = int(sys.argv[1])

a = 1
b = 1
for i in range(1,n):
    print(str(i)+': '+str(a))
    c = a + b
    a = b
    b = c

r = 1.64872
a = 1
for i in range(1,n):
    print(round(a))
    a = a* r
