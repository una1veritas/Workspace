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
    
#    def inducedEdges(self, vsub):
#        iedges = set()
#        for u, v in self.edges:
#            if u in vsub and v in vsub:
#                iedges.add( (u,v) )
#        return iedges

#引数にはコンマで区切った（tuple として解釈される）頂点の列，辺の列，
#あるいは set, tuple, list 形式での頂点の集合，辺の集合 
g = Graph(eval(sys.argv[1]), eval(sys.argv[2]))
k = int(eval(sys.argv[3]))
print('G = (V = '+str(g.vertices)+', E = '+str(g.edges)+ ' )')
print('colors = '+str(k))

#非決定的な彩色の生成をランダムでシミュレーション
coloring = { v: random.randint(0,k-1) for v in g.vertices }
print('coloring = '+str(coloring))

print('verifying...')
for edge in g.edges :
    if coloring[edge[0]] == coloring[edge[1]] :
        print('failed. '+str(edge))
        break
else:
    print('success!')
#prints out success on accept