import sys

if len(sys.argv) < 3 :
    printf('give two args.')
    exit()

a = int(sys.argv[1])
b = int(sys.argv[2])

while b != 0 :
    c = a % b
    a = b
    b = c

print("gcd = " + str(a))

