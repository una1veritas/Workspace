import sys

def modpower(x, m, n):
    t = x
    d = n
    res = 1
    while d > 0 :
        if (d & 1) :
            res = (res * t) % m
        t = (t * t) % m
        d = d>>1
    return res

def modpower1(x, m, n):
    if n == 0 :
        return 1 % m
    if n == 1 :
        return x % m
    half = int(n/2)
    t = modpower(x, m, half)
    if (n % 2) == 0 :
    #    print('.')
        return (t*t) % m
    else:
    #    print(';')
        return (t*t*x) % m

x, m, n = [int(item) for item in sys.argv[1:]]

print('x, m, n = ',x,m,n)
for i in range(1,n+1):
    print('modpower = ', i, modpower(x,m,i))
