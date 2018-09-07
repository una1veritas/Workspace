import sys

class Trie:
    ''' string word Trie '''
    ''' self.arcs[node_id] --> (dict[char] --> child_id) '''
    def __init__(self,str_set):
        self.nodes = set()
        self.arcs = dict()
        self.root = 0
        self.nodes.add(self.root)
        ''' add the children-dict when its child is added '''
        for a_word in str_set:
            self.addPath(a_word)
        
    def addPath(self, a_word):
        curr_node = self.root
        for c in a_word:
            if not (curr_node in self.arcs) :
                ''' if curr_node is leaf '''
                self.arcs[curr_node] = dict()
            if not (c in self.arcs[curr_node]) :
                new_child = len(self.nodes)
                self.nodes.add(new_child)
                self.arcs[curr_node][c] = new_child
            curr_node = self.arcs[curr_node][c]

    def traverse_preorder(self):
        path = list()
        path.append( (chr(0), self.root) )
        visiting = self.root
        while ( len(path) > 0 ):
            if ( path[-1][1] == visiting ) : # at a node never visited before
                print(path)
                if ( visiting in self.arcs) : # has some children
                    label = sorted(self.arcs[visiting].keys())[0]
                    visiting = self.arcs[visiting][label]
                    path.append( (label, visiting) )
                else: # exists no children
                    path.pop()
            else: # backed from a child
                keylist = sorted(self.arcs[path[-2][1]].keys())
                print(str(keylist)+', '+str(path[-2][0]))
                keylist = [ elem for elem in keylist if elem > path[-1][0] ]
                # print()
                '''index = 0
                for index in range(0, len(keylist) ):
                    if visiting == self.arcs[path[-1][1]][keylist[index]] :
                        break
                index = index + 1
                '''
                #if index >= len(keylist):
                if len(keylist) == 0 :
                    visiting = path[-1][1]
                    path.pop()
                else:
                    print('visiting = '+ str(visiting) + ', keylist = ' + str(keylist))
                    break;
                    nextkey = keylist[0]
                    visiting = self.arcs[path[-1][1]][nextkey]
                    path.append( (nextkey, visiting) )


    def __str__(self):
        return 'Trie' + str(self.arcs)
    

tree = Trie(['cat', 'at', 'act', 'cab', 'bat', 'tab'])
print(tree)
tree.traverse_preorder()
print('finished.')