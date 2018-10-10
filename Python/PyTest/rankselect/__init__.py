#
# -*- coding: utf-8 -*-
'''
011010001
rank 0:   1 1 1 2 2 3 4 5 5 
rank 1:   0 1 2 2 3 3 3 3 4 
select 1: 0 1 2 4 8 -1 -1 -1 -1 
fin.
'''
import sys

def rank(tstr, symb, pos):
    if pos < 0 or pos >= len(tstr) :
        return -1
    res_rank = 0
    for c in tstr[:pos+1] :
        if c == symb :
            res_rank = res_rank + 1
    return res_rank

def select(tstr, symb, rank):
    if rank < 0 :
        return -1
    pos = 0
    curr_rank = 0
    for c in tstr :
        if c == symb :
            curr_rank = curr_rank + 1
        if curr_rank == rank:
            return pos
        pos = pos + 1
    if pos == len(tstr):
        return -1
    return pos

def main():
    if len(sys.argv[1:]):
        teststr = str(sys.argv[1])
    else:
        teststr = 'test'
        
    print(teststr )
    print('rank 0: ', end='')
    for i in range(0,len(teststr)):
        print(rank(teststr,'0', i), end = '')
        print(' ', end = '')
    print()
    print('rank 1: ', end='')
    for i in range(0,len(teststr)):
        print(rank(teststr,'1', i), end = '')
        print(' ', end = '')
    print()
    print('select 1: ', end='')
    for i in range(0,len(teststr)):
        print(select(teststr,'1', i), end = '')
        print(' ', end = '')
    print()
    
    print('fin.')

main()