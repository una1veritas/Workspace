'''
Created on 2019/09/25

@author: sin
'''
import math

def decide(dset, propid, querytype = 'IDENTITY'):
    res = dict()
    for t in dset:
        if querytype == 'IDENTITY' :
            ans = t[propid]
        elif querytype == 'THRESHOLD' :
            ans = t[propid]
        if not ans in res:
            res[ans] = [t]
        else:
            res[ans].append(t)
    return res

def purity(dset):
    counters = dict()
    for elem in dset:
        if elem[-1] in counters :
            counters[elem[-1]] += 1
        else:
            counters[elem[-1]] = 1
    total = sum(counters.values())
    gain = 0  # entropy
    for a_class in counters:
        poss = counters[a_class]/total
        if poss != 0 :
            # if poss == 0 then gain += 0
            gain += -poss*math.log2(poss)
    return gain

classes = [u'台風', u'熱帯低気圧', u'温帯低気圧']
propertyname = [u'中心気圧', u'中心付近の最大風速', u'前線を伴う']
dataset = [
    [985, 32, 'no', u'熱帯低気圧'],
    [970, 45, 'no', u'台風'],
    [985, 36, 'yes', u'温帯低気圧'],
    [976, 5, 'yes', u'温帯低気圧']
    ]

for qid in range(len(dataset)-1) :
    print('\n属性 {0}（{1}）で分類した場合の情報量利得：'.format(qid, propertyname[qid]))
    result = decide(dataset, qid)
    entropy = 0
    total = 0
    for ans in result:
        print(ans, result[ans])
        total += len(result[ans])
        entropy += len(result[ans])*purity(result[ans])
    print('average information gain = ', entropy/total)