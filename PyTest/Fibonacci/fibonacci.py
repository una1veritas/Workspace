'''
Created on 2018/03/25

@author: sin
'''

def series(nth):
    if nth <= 2 : return 1
    f1 = 1
    f2 = 1
    for i in range(2,nth):
        f3 = f1 + f2
        f1 = f2
        f2 = f3
    return f3