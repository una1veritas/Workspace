import sys

class Trie:
    ''' string word Trie '''
    ''' if node_id in self.arcs then node_id is not a leaf '''
    ''' self.arcs[node_id] --> (dict[char] --> child_id) '''
    def __init__(self,str_set):
        self.nodes = set()
        self.arcs = dict()
        self.root = 0
        self.nodes.add(self.root)
        for a_word in str_set:
            self.addPath(a_word)
    def isRoot(self, node_id):
        return node_id == self.root
    
    def isLeaf(self, node_id):
        return not node_id in self.arcs
    
    def isLast(self, parent, child):
        if self.isLeaf(parent) :
            return False
        key_list = sorted(self.arcs[parent].keys())
        if len(key_list) == 0 :
            return False
        return child == self.arcs[parent][key_list[-1]]
    
    def addPath(self, a_word):
        curr_node = self.root
        for c in a_word:
            if self.isLeaf(curr_node) :
                self.arcs[curr_node] = dict()
            if not (c in self.arcs[curr_node]) :
                new_child = len(self.nodes)
                self.nodes.add(new_child)
                self.arcs[curr_node][c] = new_child
            curr_node = self.arcs[curr_node][c]

    def traverse(self):
        st_list = list()
        path = list()
        path.append( (self.root, chr(0)) )
        visiting = self.root
        while ( len(path) > 0 ):
            if  visiting == path[-1][0] : # at a node never visited before
                # visiting
                # print(str(path) + " : " + str(visiting))
                if self.isRoot(visiting) :
                    st_list.append([False, path[-1][1], False, ''])
                else:
                    t_list = [ pair[1] for pair in path ]
                    t_str = ''.join(reversed(t_list[1:-1]))
                    st_list.append( [self.isLast(path[-2][0],visiting), path[-1][1], self.isLeaf(visiting), t_str])
                if not self.isLeaf(visiting) : # has some children
                    label = sorted(self.arcs[visiting].keys())[0]
                    path.append( (self.arcs[visiting][label], label) )
                    visiting = path[-1][0]
                else: # exists no children
                    visiting = path[-2][0]
                    # print('no children')
            elif visiting == path[-2][0] :
                # backed from a leaf or an exhausted child
                last_label = path[-1][1]
                path.pop()
                key_list = [label for label in sorted(self.arcs[visiting].keys()) if label > last_label]
                if len(key_list) == 0 :
                    # exhausted children
                    if path[-1][0] == self.root :
                        break
                    visiting = path[-2][0]
                    # print('no more children')
                else:
                    # go to the next child
                    next_label = key_list[0]
                    visiting = self.arcs[visiting][next_label]
                    path.append( (visiting, next_label) )
            else:
                print('unexpected error!')
                break
        return sorted(st_list, key = lambda st: st[3])

    def __str__(self):
        return 'Trie' + str(self.arcs)
    
    
tree = Trie(['cat', 'at', 'bbc', 'act', 'cab', 'bat', 'abba'])
print(tree)
st_list = tree.traverse()
for i in range(len(st_list)):
    print(str(i) + ': ' + str(1 if st_list[i][0] else 0) + ' ' + str(st_list[i][1] if st_list[i][1] != chr(0) else ' ') + ' ' + str(1 if st_list[i][2] else 0) )
print('finished.')