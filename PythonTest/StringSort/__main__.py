import sys
from test.test_tarfile import tmpname
    
def down_to_leaf(a, start, n):
    index = start
    while index < n>>1:
        thechild = (index << 1) + 1
        if thechild + 1 < n and not (a[thechild] < a[thechild + 1]) :
            thechild = thechild + 1
        if not (a[index] < a[thechild]) :
            tmp = a[index]
            a[index] = a[thechild]
            a[thechild] = tmp
        index = thechild

def make_heap(a):
    for start in reversed(range(0, len(a)>>1)) :
        print(start)
        down_to_leaf(a, start, len(a))

array = sys.argv[1:]

print(array)
make_heap(array)
print(array)
t = array[0]
array[0] = array[-1]
array[-1] = t
down_to_leaf(array, 0, len(array)-1)
print(array)