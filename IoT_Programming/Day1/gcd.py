import sys

if len(sys.argv) < 3 :
    print("引数は整数二つです．")
    exit()

a = int(sys.argv[1])
b = int(sys.argv[2])

while b != 0 :
    c = a % b
    a = b
    b = c

print('greatest common divisor = ' + str(a) + '.')
