# import sys

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

    def xbw(self):
        st_list = list()
        path = list()
        path.append( (self.root, '$') )
        visiting = self.root
        while ( len(path) > 0 ):
            if  visiting == path[-1][0] : # on the node never visited before
                # visiting
                # print(str(path) + " : " + str(visiting))
                if self.isRoot(visiting) :
                    st_list.append([True, path[-1][1], False, ''])
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
        st_list = [t[0:3] for t in sorted(st_list, key = lambda st: st[3])]
        return st_list

#     @classmethod
#     def revxbw(self, s_list):
#         alphabet_n = set()
#         count = dict()
#         for t in s_list:
#             if t[2] :
#                 continue
#             if not t[1] in alphabet_n :
#                 alphabet_n.add(t[1])
#                 count[t[1]] = 1
#             else:
#                 count[t[1]] = count[t[1]] + 1
#         print('count=',count)
#         first = dict()
#         alph_list = sorted(alphabet_n)
#         first[alph_list[0]] = 1
#         for i in range(1,len(alph_list)) :
#             mu_count = 0
#             pos = first[alph_list[i-1]]
#             while mu_count < count[alph_list[i-1]] :
#                 # print('pos, mu_count, count = ', pos, mu_count, count[alph_list[i-1]])
#                 if s_list[pos][0] :
#                     mu_count = mu_count + 1
#                 pos = pos + 1
#             first[alph_list[i]] = pos
#         print('f=',first)
#         jump = []
#         for i in range(0,len(s_list)):
#             if s_list[i][2] :
#                 jump.append(0)
#             else:
#                 z = first[s_list[i][1]]
#                 jump.append(z)
#                 while not s_list[z][0]:
#                     z = z+ 1
#                 first[s_list[i][1]] = z + 1
#         print('jump=',jump)
#         return 

    def __str__(self):
        t_list = []
        for t_node in self.nodes:
            if t_node in self.arcs:
                t_list.append(str(t_node)+'-{')
                itempairs = sorted(self.arcs[t_node].items())
                for t_pair in itempairs[0:-1]:
                    t_list.append(str(t_pair[0]) + ' -> ' + str(t_pair[1]) + ', ')
                t_list.append(str(itempairs[-1][0]) + ' -> ' + str(itempairs[-1][1]) + '}, ')
            else:
                t_list.append(str(t_node)+', ')
        return 'Trie ' + ''.join(t_list)
    
    
class XBWTrie:
    def __init__(self, xbw_array):
        self.xbw = xbw_array
        self.alphabet_count = dict()
        for t in self.xbw:
            if t[2] :
                if not t[1] in self.alphabet_count:
                    self.alphabet_count[t[1]] = 0
            else:
                if not t[1] in self.alphabet_count :
                    self.alphabet_count[t[1]] = 1
                else:
                    self.alphabet_count[t[1]] = self.alphabet_count[t[1]] + 1
        self.firstchild = self.jumpindex()
        return
    
    def __str__(self):
        str_list = ['XBWTrie ']
        for index in range(len(self.xbw)):
            t_tuple = self.xbw[index]
            t_str = ['(', str(index)+': ']
            if t_tuple[0] :
                t_str.append('/')
            t_str.append("'"+t_tuple[1]+"'")
            if not self.xbw[index][2] :
                t_str.append('-> '+str(self.firstchild[index]))
            t_str.append('), ')
            str_list.append(''.join(t_str))
        #t_list.append(str(self.xbw)+', ')
        #str_list.append(str(self.firstchild))
        return ''.join(str_list)
    
    def isRightmost(self, index):
        return self.xbw[index][0]
    
    def inLabel(self, index):
        return self.xbw[index][1]
    
    def isLeaf(self, index):
        return self.xbw[index][2]

    def jumpindex(self):
        first = dict()
        alph_list = sorted([ a for a in self.alphabet_count if self.alphabet_count[a] != 0])
        #print('alphabet=',self.alphabet_count)
        #print('alph_list=',alph_list)
        first[alph_list[0]] = 1
        for i in range(1,len(alph_list)) :
            mu_count = 0
            pos = first[alph_list[i-1]]
            while mu_count < self.alphabet_count[alph_list[i-1]] :
                #print('pos, mu_count, count = ', pos, mu_count, count[alph_list[i-1]])
                if self.isRightmost(pos) :
                    mu_count = mu_count + 1
                pos = pos + 1
            first[alph_list[i]] = pos
        print('f=',first)
        jump = []
        for i in range(0,len(self.xbw)):
            if self.isLeaf(i):
                jump.append(0)
            else:
                z = first[self.inLabel(i)]
                jump.append(z)
                while not self.isRightmost(z):
                    z = z+ 1
                first[self.inLabel(i)] = z + 1
        return jump
            
tree = Trie(['ace', 'bag', 'beat', 'acetone', 'cat', 'coat', 'at', 'tab', 'bat', 'bad', 'cab', 'act'])
print(tree)
structure = tree.xbw()
for i in range(len(structure)):
    print(i, structure[i])
print('finished.')
xbwt = XBWTrie(structure)
print(xbwt)