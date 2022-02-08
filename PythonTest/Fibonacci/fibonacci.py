'''
Created on 2018/03/25

@author: sin
'''

def series(nth,final=None):
    if nth < 0 or (final is not None and final < nth):
        raise ValueError('unexpected arguments for fibonacci.series {}, {}.'.format(nth, final))
    if nth <= 2 : return 1
    if final is None:
        endth = nth + 1
    else:
        endth = final+1
    f1 = 1
    f2 = 1
    result = list()
    for i in range(2,endth):
        f3 = f1 + f2
        f1 = f2
        f2 = f3
        if i >= nth:
            result.append(f3)
    if final is None:
        return result.pop()   
    return result
