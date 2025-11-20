"""V=[0,1,2,3,4,5,6]
E=[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,1]]
Edgemat=[[0,1,0,0,0,0,1],
         [1,0,1,0,0,0,0],
         [0,1,0,1,0,0,0],
         [0,0,1,0,1,0,0],
         [0,0,0,1,0,1,0],
         [0,0,0,0,1,0,1],
         [1,0,0,0,0,1,0]]"""

import numpy as np
import random
from matplotlib import pyplot as plt
import networkx as nx
import time
import sys
import csv
import heapq
class Graph:
    def __init__(self,Vec,Edge):
        self.Vec=Vec.copy()
        self.Edge=Edge.copy()
        self.degree=[0 for i in self.Vec]
        self.Edgemat=np.zeros((len(self.Vec),len(self.Vec)),dtype=int)
        for [i,j] in self.Edge:
            self.Edgemat[i][j]=1
            self.Edgemat[j][i]=1
            self.degree[i]+=1
            self.degree[j]+=1
        
        
    def __str__(self):
        print("V:",self.Vec,"\n","E:",self.Edge,"\n")

    def Eremove(self,Edge):
        if Edge not in self.Edge:
            t=[Edge[1],Edge[0]]
            Edge=t
            if Edge not in self.Edge:
                print("Error:no Edge ",str(Edge))
                exit(1)
        self.Edge.remove(Edge)
        self.Edgemat[Edge[0]][Edge[1]]=0
        self.Edgemat[Edge[1]][Edge[0]]=0
        self.degree[Edge[0]]-=1
        self.degree[Edge[1]]-=1
        return 0
    def Vremove(self,v):
        if v not in self.Vec:
            print("Error:no Edge ",str(v))
            exit(1)
        self.Vec.remove(v)
        self.degree[v]=0
        return 0

    def Neighborlist(self,v):#vの隣接点リストを返す
        return [x for x in self.Vec if self.Edgemat[v][x]==1 ]
    def n_Neighborlist(self,v,n):
        if n==1:
            return self.Neighborlist(v)
        r=[]
        for i in self.n_Neighborlist(v,n-1):
            r.extend(self.Neighborlist(i))
        return r
        
    def InEdge(self,v):#vが入っている辺リストを返す   
        return [[v,y] for y in self.Vec if self.Edgemat[v][y]==1 or self.Edgemat[y][v]==1]
    def NeighborEdge(self,v):#vの近隣辺リストを返す
        return [[x,y] for [x,y] in self.Edge if x in self.Neighborlist(v) and y in self.Neighborlist(v)]

        
    def MaxNeighborV(self):#最大隣接点辺の点を返す
        #隣接辺数リスト作成
        heapdegree=self.degree.copy()
        heapdegree=[-1*i for i in heapdegree]
        heapq.heapify(heapdegree)
        maxdegree=heapq.heappop(heapdegree)
        
        for i in range(len(self.degree)):
            if self.degree[i]==-maxdegree:
                return i#隣接点最大の頂点を返す




class Greedy:

    def GreedyHubCover(V,E,f):
        mnvcount=0
        Ecount=0
        setH=[]
        log_buffer=[]
        remainedG=Graph(V,E)
        while len(remainedG.Edge)>0:#辺が残っている限り
            MNV=remainedG.MaxNeighborV()
            mnvcount+=1
            setH.append(MNV)
            #print("MNV:"+str(MNV))
            log_buffer.append(f"{mnvcount} MNV={MNV}\n")
            #近隣辺削除
            for i in remainedG.NeighborEdge(MNV):
                remainedG.Eremove(i)
                Ecount+=1
                f.writelines([str(Ecount),"del ",str(i),"\n"])
                #print("del "+str(i))
            #隣接辺削除
            for i in remainedG.InEdge(MNV):
                remainedG.Eremove(i)
                Ecount+=1
                f.writelines([str(Ecount),"del ",str(i),"\n"])
                #print("del "+str(i))
            remainedG.Vremove(MNV)
            f.writelines(["del ",str(MNV),"\n"])
            #print("del "+str(MNV))

        return setH

def main():
    if len(sys.argv)<=2 or len(sys.argv)>=4:
        print("Error:Study3.py 頂点数 辺数")
        exit(1)
    Vecnum=int(sys.argv[1])
    Edgenum=int(sys.argv[2])
    dataf=open("Data.txt","a")
    f=open("GreedyHubCoverlog"+str(Vecnum)+".txt","w")
    V=[i for i in range(0,Vecnum)]
    E=[]
    Edgemat=np.zeros((len(V),len(V)),dtype=int)
    Enum=0
    while Enum<Edgenum:
        u=random.randint(0,Vecnum-1)
        v=random.randint(0,Vecnum-1)
        if u==v:
            continue
        if Edgemat[u][v]==0 and Edgemat[v][u]==0:
            Edgemat[u][v]=1
            Edgemat[v][u]=1
            E.append([u,v])
            Enum+=1
    start=time.time()
    H=Greedy.GreedyHubCover(V,E,f)
    end=time.time()
    print("Greedytime:",end-start,"秒")
    print("Greedy H=",len(H))
    #隣接解を求める
    startna=time.time()
    G=Graph(V,E)
    while 1:
        caltimelist=[]
        improved=0
        #notH=[v for v in V if v not in H]
        for i in V:
            start=time.time()
            preH=[]
            #隣接点と近隣点をHから除く
            preH=[v for v in H if v not in G.Neighborlist(i) and v not in G.n_Neighborlist(i,2)]
            #iを追加する
            if i not in H:
                preH.append(i)
            #Greedy法でhubcoverする
            rap1=time.time()
            NotCoverE=[]
            CoverEmat=np.zeros((Vecnum,Vecnum),dtype=int)
            for k in preH:
                for p in G.InEdge(k)+G.NeighborEdge(k):
                    CoverEmat[p[0]][p[1]]=1
                    CoverEmat[p[1]][p[0]]=1
            for k in E:
                if CoverEmat[k[0]][k[1]]==0:
                    NotCoverE.append(k)
            preH.extend(Greedy.GreedyHubCover(V,NotCoverE,f))
            if len(preH)<len(H):
                H=preH
                improved=1
                print("improve success")
                break
            end=time.time()
            caltimelist.append(end-start)
        if improved==0:break

    print("last H=",len(H))
    print("nacaltime",time.time()-startna,"秒")
    print("average_1na_caltime",sum(caltimelist)/Vecnum)
    # CSVファイルを作成
    with open('data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # データを書き込む
        writer.writerow([Vecnum,Edgenum])
    G=nx.DiGraph()
    edges=E
    G.add_edges_from(edges)
    #nx.draw(G,with_labels=True,node_color="lightblue",node_size=500,font_size=10,arrows=False)
    #plt.show()

if __name__=="__main__":
    main()