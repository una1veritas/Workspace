<<<<<<< HEAD
import sys

def ystring(xstr):
    return ''.join([ c if c == '0' else '10' for c in xstr])

def select(tstr, c, num):
    count = -1
    for idx in range(0, len(tstr)) :
        if tstr[idx] == c :
            count = count + 1
        if count == num :
            return idx
    else:
        return -1

def rank(tstr, c, pos):
    count = 0
    for t in tstr[:pos+1] :
        if t == c :
            count = count + 1
    return count
    
def main():
    xstr = sys.argv[1]
    ystr = ystring(xstr)
    print(xstr, ystr)
    
    select_Y = ([],[])
    for j in range(0,len(ystr)):
        l = select(ystr, '0', j)
        if l == -1 :
            break
        select_Y[0].append(l) 
    for j in range(0,len(ystr)):
        l = select(ystr, '1', j)
        if l == -1 :
            break
        select_Y[1].append(l) 
        
    print('rank_X(0,j) = ', [rank(xstr,'0',j) for j in range(0,len(xstr))])
    print('rank_X(1,j) = ', [rank(xstr,'1',j) for j in range(0,len(xstr))])
    print('select_X(0,j) = ', [select(xstr,'0',j) for j in range(0,len(xstr))])
    print('select_X(1,j) = ', [select(xstr,'1',j) for j in range(0,len(xstr))])
    print()
    print('select_Y(0,j) = ', select_Y[0])
    print('select_Y(1,j) = ', select_Y[1])
    print()
    print('2j + 1 - select_Y(0,j) = ', [ (2*j + 1- select_Y[0][j]) for j in range(0,len(xstr))])
    print(' = rank_X(0,j)')
    print('select_Y(0,j) - j = ', [ (select_Y[0][j] - j) for j in range(0,len(xstr))])
    print(' = rank_X(1,j)')
    print('select_Y(1,j) - j = ', [ (select_Y[1][j] - j) for j in range(0,len(select_Y[1]))])
    print(' = select_X(1,j)')
=======
#
# -*- coding: utf-8 -*-
'''
011010001 in {'0', '1'}^*

rank 0:    1 1 1 2 2 3 4 5 5 
rank 1:    0 1 2 2 3 3 3 3 4 

select 0: 0 3 5 6 7 -1 -1 -1 -1 
select 1: 1 2 4 8 -1 -1 -1 -1 -1 
done.
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

def select(tstr, symb, ith):
    if ith < 0 :
        return -1
    pos = 0
    curr_rank = -1
    for c in tstr :
        if c == symb :
            curr_rank = curr_rank + 1
        if curr_rank == ith:
            return pos
        pos = pos + 1
    else:
        return -1
    return pos

def main():
    if len(sys.argv[1:]):
        teststr = str(sys.argv[1])
    else:
        teststr = 'test'

    alphabet = set([c for c in teststr])
    print(teststr+' in '+str(alphabet)+'^*' )
    
    ''' for X = teststr '''
    for c in alphabet:
        print('rank {symb}:\t'.format(symb=c), end='')
        print( [ rank(teststr, c, i) for i in range(0, len(teststr)) ])
    print()
    for c in alphabet:
        print('select {symb}: '.format(symb=c), end='')
        for i in range(0,len(teststr)):
            print(select(teststr,c, i), end = '')
            print(' ', end = '')
        print()
    
    print()
    ''' for Y '''
    teststr = ''.join([c if c == '0' else '10' for c in teststr])
    print(teststr)
    for c in alphabet:
        print('rank {symb}:\t'.format(symb=c), end='')
        for i in range(0,len(teststr)):
            print(rank(teststr,c, i), end = '')
            print(' ', end = '')
        print()

    print()
    for c in alphabet:
        print('select {symb}: '.format(symb=c), end='')
        for i in range(0,len(teststr)):
            print(select(teststr,c, i), end = '')
            print(' ', end = '')
        print()
    
    print('done.')
>>>>>>> cee2f6443c2c9f2dc00566dbf0317e625b9bab7f

main()