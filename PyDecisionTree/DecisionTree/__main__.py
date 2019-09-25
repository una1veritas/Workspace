'''
Created on 2019/09/25

@author: sin
'''
import math

def decide(qid, dset):
    res = dict()
    for t in dset:
        ans = t[qid]
        if not ans in res:
            res[ans] = [t]
        else:
            res[ans].append(t)
    return res

def infval(possibility):
    if possibility == 0 :
        return 0
    return -math.log2(possibility)

def purity(dset):
    countdict = dict()
    for elem in dset:
        if elem[-1] in countdict :
            countdict[elem[-1]] += 1
        else:
            countdict[elem[-1]] = 1
    numclasses = len(countdict)
    numexamples = sum(countdict.values())
    gain = 0
    for a_class in countdict:
        poss = countdict[a_class]/numexamples
        gain += poss*infval(poss)
    return gain

classes = [u'台風', u'熱帯低気圧', u'温帯低気圧']
queries = ['中心気圧', '中心付近の最大風速（ノット）', '前線を伴う', 'class']

dataset = [
    [985, 32, 'no', u'熱帯低気圧'],
    [990, 45, 'no', u'台風'],
    [985, 36, 'yes', u'温帯低気圧']
    ]

questionid = 1
print('\n質問 {0} で分類した場合の情報量利得：'.format(questionid))
result = decide(questionid,dataset)
entropy = 0
total = 0
for ans in result:
    print(ans, result[ans])
    total += len(result[ans])
    entropy += len(result[ans])*purity(result[ans])
print('average information gain = ', entropy/total)