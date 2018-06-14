import sys
import random

vertices = set()
for v in sys.argv[1].split():
    vertices.add(v)
elist = sys.argv[2].split()
edges = set()
for i in range(0, len(elist), 2) :
    edges.add( tuple(sorted([elist[i], elist[i+1]])) )

print('Vertices = '+str(vertices))
print('Edges = '+str(edges))

circuit = list(vertices)
#非決定的な順列の生成をランダムシャッフルでシミュレーション
random.shuffle(circuit)

print('circuit = '+str(circuit))

print('checking the circuit...')
for i in range(0, len(vertices)) :
    j = (i + 1) % len(vertices)
    a_pair = (circuit[i], circuit[j])
    print( a_pair )
    a_pair = tuple(sorted(list(a_pair)))
    if not (a_pair in edges) :
        print('failed at '+str(a_pair))
        break
else:
    print('success!')