import sys
import random

class Graph():
    def __init__(self,v,e):
        self.vertices = set(v)
        self.edges = set()
        for u, v in e:
            if u != v : #rejects self loop
                edge = (u,v) if u < v else (v,u)
                self.edges.add(edge)

    def adjacent(self, u, v):
        if u < v :
            return (u,v) in self.edges
        else:
            return (v,u) in self.edges 

#引数にはコンマで区切った（tuple として解釈される）頂点の列，辺の列，
#あるいは set, tuple, list 形式での頂点の集合，辺の集合 
g = Graph(eval(sys.argv[1]), eval(sys.argv[2]))
print('G = (V = '+str(g.vertices)+', E = '+str(g.edges)+ ' )')
circuit = list(g.vertices)

#非決定的な順列の生成をランダムシャッフルでシミュレーション
random.shuffle(circuit)
print('circuit = '+str(circuit))

print('checking the circuit...')
for i in range(0, len(g.vertices)) :
    j = (i + 1) % len(g.vertices)
    print(str(circuit[i])+' -> '+str(circuit[j])+', ', end='')
    if not g.adjacent(circuit[i], circuit[j]) :
        print('failed.')
        break
    print()
else:
    print('success!')
#prints out success on accept