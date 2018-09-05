import sys

class Trie:
    ''' string word Trie '''
    def __init__(self,str_set):
        self.nodes = set()
        self.arcs = dict()
        self.root = 0
        self.nodes.add(self.root)
        for a_word in str_set:
            self.addPath(a_word)
        
    def addPath(self, a_word):
        curr_node = self.root
        for c in a_word:
            if not ( (curr_node, c) in self.arcs) :
                ''' new node == the last node id + 1 == the size '''
                new_node = len(self.nodes)
                self.nodes.add(new_node)
                self.arcs[(curr_node,c)] = new_node 
            curr_node = self.arcs[(curr_node,c)]
            
    def __str__(self):
        return 'Trie' + str(self.arcs)
    

tree = Trie(['cat', 'at', 'act', 'cab', 'bat', 'bcc', 'tab'])
print(tree)