import sys

def gcd(a, b):
    while b != 0 :
        c = a % b
        a = b
        b = c
    return a

if len(sys.argv) < 3 :
    print("引数は整数二つです．")
    exit()

x = int(sys.argv[1])
y = int(sys.argv[2])
print(x, 'と', y, 'の比は')

c = gcd(x, y)

print(x//c, '対', y//c, 'です')   # // は整数除算（あまりは捨てる）
