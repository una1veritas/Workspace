import sys

print('引数 = ', sys.argv)
r = float(sys.argv[1])
sq = r**2

print(str(r)+' の二乗は '+str(sq))

print('プログラム中で標準入力をうけとる')
s = input('数値を入力 ')
print('三乗は '+str(float(s)**3))

l = [1,2,3]
print(type(l))
t = tuple(l)
print(isinstance(t,tuple))
