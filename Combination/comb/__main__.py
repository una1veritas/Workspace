
import itertools

l = [i for i in range(1,13)]

print(l)
c = 0
for perm in itertools.permutations(l, 12):
    c += 1
print(c)