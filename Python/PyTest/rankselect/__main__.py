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

main()