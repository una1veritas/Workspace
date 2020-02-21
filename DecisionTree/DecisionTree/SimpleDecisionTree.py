#
#
import math
import MeCab
from graphviz import Digraph
import os

class DecisionTree:
    label = None
    children = None
    qtype = None
    
    def __init__(self, labelObj=None, childs=None, qtype = None):
        self.label = labelObj
        self.children = childs
        self.qtype = None
    
    def makeDecisionTree(self, database, selections, testColumn, targetColumn, queryType='regex'):
        if self.data_is_pure(database, selections, targetColumn) :
            self.label = database[selections[0]][targetColumn]
            self.children = None
            self.qtype = queryType
            return
        #print(selections, testColumn, targetColumn)
        if queryType == 'regex' :
            wordset = self.collect_substrings(database, selections, testColumn)
            (word, decisions) = self.choose_substring(database, selections, wordset, testColumn, targetColumn)
            self.label = word
            self.qtype = queryType
        elif queryType == 'analyzedword' :
            wordset = self.collect_analyzedwords(database, selections, testColumn)
            (word, decisions) = self.choose_analyzedword(database, selections, wordset, testColumn, targetColumn)
            self.label = word
            self.qtype = queryType
            #print('error: type \''+str(queryType)+'\' is still not supported.')
        else: 
            print('error: type \''+str(queryType)+'\' is still not supported.')
            return
        self.children = dict()
        for key in decisions :
            self.children[key] = DecisionTree()
            self.children[key].makeDecisionTree(database, decisions[key], testColumn, targetColumn, queryType)
        return
        
    def label_string(self):
        ostr = ''
        if self.is_empty() :
            return 'DecisionTree'
        if self.qtype == 'regex' :
            ostr = str(self.label)
        elif self.qtype == 'analyzedword' :
            if self.is_leaf() :
                ostr = self.label
            else:
                ostr = self.label[0]
                for item in self.label[1:]:
                    ostr += '_'
#                    if item != '*':
                    ostr += item
        else:
            ostr = 'unknowntype: ' + str(self.label)
        return ostr
        
    def __str__(self):
        ostr = self.label_string()
        if self.is_leaf() or len(self.children) == 0 :
            return ostr
        ostr += '['
        is_first_elem = True
        for key, val in sorted(self.children.items()):
            if not is_first_elem :
                ostr += ', '
            ostr += str(key) + '-> ' + val.__str__()
            is_first_elem = False
#         path = [self]
#         while len(path):
#             for a in sorted(path[-1].children.keys()):
#                 ostr += path[-1].children[a].__str__() + ', '
#             path.pop(-1)
        ostr += ']'
        return ostr
    
    def data_is_pure(self, database, indices, targetColumn):
        target_classes = set()
        for idx in indices:
            target_classes.add(database[idx][targetColumn])
        return len(target_classes) == 1
    
    def collect_substrings(self, databases, selections, textIndex):
        words = set()
        for idx in selections:
            a_line = databases[idx][textIndex]
            if len(a_line) == 0 :
                continue
            for a_word in [ a_line[i:j] for i in range(0, len(a_line)) for j in range(i+1, len(a_line))]:
                words.add(a_word)
        return words

    def collect_analyzedwords(self, database, selections, analyzedIndex):
        words = set()
        #tagger = MeCab.Tagger("-Ochasen")
        for idx in selections:
            sentence = database[idx][analyzedIndex]
            for a_word in sentence :
                words.add( a_word )
                wildword = ('*', a_word[1], a_word[2])
                words.add( wildword )
        return words
        
    def choose_analyzedword(self, database, selections, words, textColumn, targetColumn):
        (bestword, bestgain, bestdecision) = ('', 0, None)
        for a_word in words:
            decision = self.classify_by_analyzedword(a_word, database, selections, textColumn, targetColumn)
            val = self.info_gain(database, decision, targetColumn)
            #print(a_word, val, decision)
            if bestgain < val or (bestgain == val and (a_word[0] != '*' and bestword[0] == '*') ):
                bestgain = val
                bestword = a_word
                bestdecision = decision
        #print('best = '+str( (bestword, bestdecision) ))
        return (bestword, bestdecision)
    
    def choose_substring(self, database, selections, words, textColumn, targetColumn):
        #if len(target_classes) == 1 :
        #    print('choose_simpleregx: error, "Already uniquely classified."')
        #    return ('', None)
        (bestword, bestgain, bestdecision) = ('', 0, None)
        for a_word in words:
            decision = self.classify_by_simpleregx(a_word, database, selections, textColumn, targetColumn)
            val = self.info_gain(database, decision, targetColumn)
            #print(a_word, val, decision)
            if bestgain < val or (bestgain == val and len(bestword) < len(a_word) ):
                bestgain = val
                bestword = a_word
                bestdecision = decision
            #print('-----')
        #print(bestword, bestdecision)
        return (bestword, bestdecision)
    
    def classify_by_simpleregx(self, labelobj, database, selections, testColumn, targetColumn):
        res = dict()
        for idx in selections:
            ans = labelobj in database[idx][testColumn]
            #print(labelobj, ans, rec[propertyIndex], rec[targetIndex])
            if ans not in res:
                res[ans] = list()
            res[ans].append( idx )
        return res
    
    def classify_by_analyzedword(self, labelobj, database, selections, analyzedIndexColumn, targetColumn):
        res = dict()
        for idx in selections:
#            ans = labelobj in database[idx][analyzedIndexColumn]
            for w in database[idx][analyzedIndexColumn] :
                for p in zip(labelobj, w) :
                    if p[0] == '*' or p[1] == '*' :
                        continue
                    if p[0] != p[1] :
                        break
                else:
                    ans = True
                    break
            else:
                ans = False
            #print(labelobj, ans, rec[propertyIndex], rec[targetIndex])
            if ans not in res:
                res[ans] = list()
            res[ans].append( idx )
        return res
    
    def info_gain(self, database, decisions, targetColumn):
        total = sum([ len(val) for key, val in decisions.items()])
        info = 0
        for a_decision, selections in decisions.items():
            decision_entropy = 0
            target_class = set([ database[idx][targetColumn] for idx in selections])
            for a_class in target_class:
                prob = len([idx for idx in selections if database[idx][targetColumn] == a_class])/len(selections) if len(selections) > 0 else 0
                decision_entropy +=  - prob * math.log(prob) if prob > 0 else 0 
            #print(ans_entropy)
            total += len(selections)
            info += len(selections) * decision_entropy
        info = info/total
        #print(info, 1 - info)
        return 1 - info
    
    def is_leaf(self):
        return self.children == None or len(self.children) == 0
    
    def is_empty(self):
        return self.label == None

    def graphdef(self):
        nodes = list()
        edges = list()
        path = [('root.', self)]
        visitp = path[-1]
        while path:
            if visitp[1] == path[-1][1] :
                nodes.append(visitp[1])
                #print(visitp[1].label_string())
            if path[-1][1].is_leaf() :
                #print('# reached to a leaf')
                path.pop()
                #print('top='+str(path[-1][0])+'->'+path[-1][1].label_string())
                #print('visit='+str(visitp[0])+'->'+visitp[1].label_string())
                continue
            else:
                if visitp[1] != path[-1][1] :
                    # backed from a child
                    # then check the next sibling for visitp[1] exists
                    lastlabel = visitp[0]
                    nextchild = False
                    for label in sorted(path[-1][1].children.keys()):
                        if nextchild :
                            path.append( (label, path[-1][1].children[label]) )
                            #print('# next label = ' + str(label) )
                            visitp = path[-1]
                            #print('# go to the next')
                            edges.append( (path[-2][1], path[-1][1], path[-1][0]) )
                            break
                        if label == lastlabel :
                            nextchild = True
                    else:
                        #print('# last child ' + str(lastlabel) )
                        visitp = path.pop()
                        #print('# go to upward')
                    continue
                else:
                    # down to the 1st child.
                    label = sorted(visitp[1].children.keys())[0]
                    child = visitp[1].children[label]
                    path.append( (label, child) )
                    visitp = (label, child)
                    edges.append( (path[-2][1], path[-1][1], path[-1][0]) )
                    #print('# going down')
                    continue
        return (nodes, edges)
        
    def graphdef_r(self, nodes, edges):
        nodes.append(self.label_string())
        for key, value in self.children.items():
            edges.append( (self.label_string(), value.label_string(), key) )
        for a_child in self.children.values() :
            if not a_child.is_leaf() :
                a_child.graphdef_r(nodes, edges)
        return (nodes, edges)
    
    def graph(self):
        g = Digraph(format='pdf')
        g.attr('graph', bgcolor='#f7f7f7', fontsize='18')
        g.attr('node', fillcolor = "white", shape='box' )
        (nodes, edges) = self.graphdef()
        for node in nodes:
            if node.is_leaf() :
                g.node(node.label_string(), style='solid,filled,rounded', shape='box')
            else:
                g.node(node.label_string(), style='solid,filled', shape='box')
        for edge in edges:
            g.edge(edge[0].label_string(), edge[1].label_string(), label=str('ある' if edge[2] else 'ない'), arrowhead = 'normal')
        return g
        
    def dot_script(self):
        header = """digraph graph_name {{
  graph [
    charset = "UTF-8";
    label = "{0}",
    bgcolor = "#f0f0f0",
  ];

    node [
      colorscheme = "white"
      style = "solid,filled",
      fillcolor = "white",
    ];

    edge [
      style = solid,
      fontsize = 18,
      fontcolor = black,
      color = black,
      labelfloat = true,
    ];
"""
        footer = ' }'
        
        nodes = list()
        edges = list()
        self.graphdef(nodes, edges)
        nodestr = '  // node definitions\n'
        for a_node in nodes :
            nodestr += '  {0} [shape = box];\n'.format(str(a_node))
        edgestr = '  // edge definitions\n'
        for an_edge in edges :
            edgestr += '  {0} -> {1} [label = "{2}", arrowhead = normal];\n'.format(an_edge[0], an_edge[1], an_edge[2])
        return header.format('DecisionTree')+nodestr+edgestr+footer


#program begins

data_table = []
with open('./patient.csv') as dbfile:
    idx = 0;
    for a_line in dbfile.readlines() :
        fields = a_line.split(',')
        fields = [ item.strip() for item in fields]
        fields = [idx] + fields
        data_table.append( tuple(fields) )
        idx += 1
[print(r) for r in data_table[:4] + ['...', '\n']]

if True:
    textIndex = [2]
    tagger_opt = '-Ochasen'
    #tagger = MeCab.Tagger("-Ochasen")
    tagger = MeCab.Tagger(tagger_opt)
    for idx in range(0, len(data_table)):
        a_text = ' '.join([data_table[idx][index] for index in textIndex])
        node = tagger.parseToNode(a_text)
        a_list = list()
        while node:
            word = node.surface
            wordinfo = node.feature.split(',')
            if wordinfo[0] != u'BOS/EOS':
                if tagger_opt == '-Ochasen' :
                    a_list.append( (word, wordinfo[0], wordinfo[1]) )
                elif tagger_opt == '-Owakati':
                    a_list.append( word )
            node = node.next
        newrecord = list(data_table[idx])
        if tagger_opt == '-Ochasen' :
            newrecord.append(a_list)
        elif tagger_opt == '-Owakati':
            a_text = ' '.join(a_list)
            newrecord.append(a_text)
        data_table[idx] = tuple(newrecord)
    [print(r) for r in data_table[:3] + ['...', '\n'] ]

dtree = DecisionTree()
dtree.makeDecisionTree(data_table, range(0, len(data_table)), 4, 3, 'analyzedword')
#print(dtree)

print('Result: ')
if '/opt/local/bin' not in os.environ['PATH']:
    os.environ['PATH'] += ':/opt/local/bin'
dtree.graph().view()
#print(dtree.graphviz())
#with open('patient.dot', mode='w') as wfile:
#    wfile.write(dtree.dot_script())
