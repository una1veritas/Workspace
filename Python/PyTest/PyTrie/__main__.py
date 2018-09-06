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

    def traverse(self):
        path = list()
        path.append( (chr(0), self.root) )
        print(path)
        while ( len(path) > 0 ):
            curr_label, curr_node = path[-1]
            if ( curr_node in self.arcs ) : # has some children 
                next_label = list(self.arcs[curr_node].keys())[0]
                next_node = self.arcs[curr_node][next_label]
                path.append( (next_label, next_node) )
                print(path)
            else:
                # reached to a leaf.
                print('a leaf.')
                break
            
    def __str__(self):
        return 'Trie' + str(self.arcs)
    

tree = Trie(['cat', 'at', 'act', 'cab', 'bat', 'bcc', 'tab'])
print(tree)
tree.traverse()
print('finished.')