'''
Created on 2018/09/02

@author: Sin Shimozono
'''
class Trie():
    
    def __init__(self,words):
        self.nodes = set()
        self.edges = dict()
        self.root = 0
        self.nodes.add(self.root)
        for a_word in words:
            self.add(a_word)
        

    def add(self, a_word):
        curr_node = self.root
        for c in a_word:
            if not (curr_node, c) in self.edges :
                self.edges[(curr_node, c)] = len(self.nodes) + 1
            curr_node = self.edges[(curr_node, c)]
        
    def __str__(self):
        result = 'Trie' + str(self.edges)
        return result

t = Trie(['this', 'it', 'that'])
print(t)