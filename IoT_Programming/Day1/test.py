import sys

if len(sys.argv) < 3 :
    print("引数は整数二つです.")
    exit()

x = int(sys.argv[1])
y = int(sys.argv[2])
print(x, ’ と’, y, ’ の比は

a, b = x, y
while b != 0 :
    c=a%b
    a=b
    b=c

print(x//a, y//a)
