'''
Created on 2025/10/18

@author: sin
'''
import pandas as pd
import numpy as np
import math
import itertools

def divide_by_property(tbl, col_name):
    tix = tbl[0].index(col_name) # raise an error if not appears in column name
    val2rows = dict()
    for rix in range(1, len(tbl[1:])):
        propval = tbl[rix][tix]
        if propval not in val2rows : val2rows[propval] = list()
        val2rows[propval].append(rix)
    return val2rows

def class_purity(tbl, col_name):
    val2rows = divide_by_property(tbl, col_name)
    return

def best_query(df):
    best_gain = 0.0
    best_col = None
    decisions = df.columns[-1], list(df[df.columns[-1]].value_counts().index)
    for col in df.columns[:-1]: 
        '''compute inf gain for col'''
        col_val_count = df[col].value_counts()
        total_rows = col_val_count.sum()
        proddict = dict()
        for val in col_val_count.keys():
            proddict[val] = dict()
            for dname in decisions[1]:
                proddict[val][dname] = len(df[(df[col] == val) & (df[decisions[0]] == dname)])
        print(col, proddict)            
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
        
            
            
if __name__ == '__main__':
    df = pd.read_csv('golf-dataset.csv')
    query, infgain = best_query(df)
    exit(1)
    dflist = separate_by_query(df, query)
    for each in dflist: print(each)
    