'''
Created on 2025/10/18

@author: sin
'''
import math

def read_csv(fname):
    tbl = list()
    with open(fname, "r") as file:
        for each in file:
            row = each.strip().split(',')
            tbl.append(row)
    return tbl

def property_values(tbl, col_name):
    tix = -1
    for cix in range(len(tbl[0])):
        if tbl[0][cix] == col_name :
            tix = cix
            break
    if tix == -1 : raise ValueError(f'No such a property: {col_name}')
    valcounts = dict()
    for each in tbl[1:]:
        val = each[tix]
        if val not in valcounts : valcounts[val] = 0
        valcounts[val] += 1
    return valcounts

def best_query(tbl):
    クラスをつかってないやん
    best_gain = 0.0
    best_col = None
    for col in tbl[0][:-1]: # for each column title
        d = property_values(tbl, col)
        total = sum(d.values())
        gain = 0.0
        for pval, count in d.items():
            gain += -count/total * math.log(2, count/total) if count > 0 else 0
        if not best_col or (best_col and best_gain < gain) :
            best_col = col
            best_gain = gain
    
    return best_col, best_gain            
    
def separate_by_query(tbl, query):
    tbldict = dict()
    header = tbl[0]
    qix = header.index(query) # throws error if there is no such column title
    for row in tbl[1:]:
        ans = row[qix]
        if ans in tbldict:
            tbldict[ans].append(row)
        else:
            tbldict[ans] = [header, row]
    return tbldict.values()

class DecisionTree:
    class QueryNode:
        def __init__(self, query, answers):
            self.query = query
            self.childrens = [(ans, None) for ans in answers]
        
    class ClassNode:
        def __init__(self, label):
            self.class_name = label
        
            
    def __init__(self, tbl):
        query, infgain = best_query(tbl)
        tbls = separate_by_query(tbl, query)
        if len(tbls) > 1 :
            self.root = QueryNode(query, tbls.keys())
        else:
            
            
if __name__ == '__main__':
    tbl = read_csv('golf-dataset.csv')
    query, infgain = best_query(tbl)
    tbls = separate_by_query(tbl, query)
    print(tbls)