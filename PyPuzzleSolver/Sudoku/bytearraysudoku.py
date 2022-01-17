#
#
import math, copy
from array import array
import datetime

class SudokuSolver():
    def __init__(self, arg):
        if isinstance(arg,(list,str)) and len(arg) in [16,81,256]:
            self.size = int(math.sqrt(len(arg)))
            self.array = bytearray( (len(arg)+1)>>1 )
            for r in range(self.size):
                for c in range(self.size):
                    self.put(r,c,int(arg[self.size*r+c]))
        elif isinstance(arg,bytearray) and len(arg) in [8,41,128]:
            self.size = int(math.sqrt(len(arg)*2))
            self.array = copy.copy(arg)
        elif isinstance(arg,SudokuSolver) :
            self.size = arg.size
            self.array = copy.copy(arg.array)
        else:
            raise ValueError('argument array has illegal size = {}'.format(len(arg)))            

    
    # def bits(self, num):
    #     if num == 0 : 
    #         return (1<<self.size) - 1
    #     if num == -1 :
    #         return 0
    #     return 1<<(num-1)
    #
    # def debits(self, bval):
    #     if self.size == 9 :
    #         if bval == 0 : return -1
    #         if bval == (1<<self.size) - 1 :return 0
    #         val = 1
    #         while bval != 0 and (bval & 1) == 0:
    #             val += 1
    #             bval >>= 1
    #         if (bval>>1) == 0 :
    #             return val
    #     elif self.size == 16:
    #         pass
    #     elif self.size == 4:
    #         pass
    #     return 0
    #
    # def _popcount(self,val):
    #     pcntbl = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
    #     count = 0
    #     while val != 0 :
    #         count += pcntbl[val & 0x0f]
    #         val >>= 4
    #     return count
    #
    # def row(self,i):
    #     return i // self.size
    #
    # def col(self,i):
    #     return i % self.size
    #
    # def index(self,row,col):
    #     return row*self.size + col
    #
    # def put(self,row,col,num):
    #     if num == 0 :
    #         return True
    #     #print(bin(self.bits_at(row, col)),num)
    #     if self.bits_at(row, col) & self.bits(num) == 0 :
    #         raise RuntimeError('Illegal placement of number {} at {},{}'.format(num,row,col))

    def make_allowedmap(self):
        onetonine = [n for n in range(1,10)]
        allowed =  [set(onetonine) if self.index_at(i) == 0 else set([self.index_at(i)]) for i in range(self.size**2)]
        for row in range(self.size):
            for col in range(self.size):
                num = self.at(row,col)
                if num != 0 :
                    for r,c in self.relatecells(row, col):
                        if r == row and c == col :
                            continue
                        allowed[r*self.size+c].discard(num)
        #print("allowed = ",allowed)
        return allowed
    
    # def bits_at(self,row,col):
    #     return self.array[row*self.size+col]
    
    def index_at(self,ix):
        md = ix & 1
        if md == 0 :
            return self.array[ix>>1] & 0x0f
        else:
            return (self.array[ix>>1]>>4) & 0x0f
        
    def at(self,row,col):
        ix = row*self.size+col
        if (ix & 1) == 0 :
            return self.array[ix>>1] & 0x0f
        else:
            return (self.array[ix>>1]>>4) & 0x0f
        
    def put(self,row,col,num):
        ix = row*self.size+col
        #print(row,col,ix,num,ix>>1, ix&1,num if ix&1 == 0 else num<<4)
        if (ix & 1) == 0 :
            self.array[ix>>1] &= 0xf0
            self.array[ix>>1] |= 0x0f & num
        else:
            self.array[ix>>1] &= 0x0f
            self.array[ix>>1] |= 0xf0 & (num<<4)

    def signature(self):
        return self.array
    
    def __str__(self):
        #print(self.size,len(self.array))
        factor = int(math.sqrt(self.size))
        tmp = ''
        for r in range(self.size):
            for c in range(self.size):
                tmp += str(self.at(r, c) if self.at(r, c) != 0 else ' ')
                if c % factor == factor - 1:
                    tmp += '|'
                else:
                    tmp += ' '
            tmp += '\n'
            if r % factor == factor - 1 :
                tmp += '-----+-----+-----+\n'
        return tmp
    
    def __hash__(self):
        hashval = 0
        factor = {4:2, 9:3, 16:4}[self.size]
        for val in self.array:
            hashval = (hashval<<factor) ^ val
        return hashval
    
    def __eq__(self, another):
        return self.array == another.array 
    

    # def at(self, row, col):
    #     return self.cells[self.index(row,col)]
    #
    # def put(self, row, col, num):
    #     self.cells[self.index(row,col)] = num
    #
    # def index(self, row, col):
    #     return row*self.size+col
    #
    def issolved(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.at(r,c) == 0 :
                    return False
        return True
    
    def relatecells(self,row,col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        factor = int(math.sqrt(self.size))
        baserow = (row // factor)*factor
        basecol = (col // factor)*factor
        for r in range(baserow, baserow+factor):
            for c in range(basecol, basecol+factor):
                yield (r, c)
        for c in range(0,basecol,1):
            yield (row, c)
        for c in range(basecol+factor,self.size,1):
            yield (row, c)
        for r in range(0,baserow,1):
            yield (r, col)      
        for r in range(baserow+factor,self.size,1):
            yield (r, col)
    
    def tighten(self, allowed):
        places = set([(r,c) for r in range(self.size) for c in range(self.size)])
        while bool(places):
            (row, col) = places.pop()
            tsize = len(allowed[row*self.size+col]) 
            if tsize == 0 :
                return False
            elif tsize == 1 :
                for e in allowed[row*self.size+col]: 
                    num = e
                    break
            else:
                continue
            for r,c in self.relatecells(row, col) :
                if num not in allowed[r*self.size+c]:
                    continue
                if row == r and col == c :
                    continue
                allowed[r*self.size+c].discard(num)
                sizeafter = len(allowed[r*self.size+c])
                if sizeafter == 0 :
                    #print(r,c,num)
                    return False 
                if sizeafter == 1 :
                    for e in allowed[r*self.size+c]:
                        n = e
                        break
                    places.add( (r,c) )
                    self.put(r,c,n)
        return True
    
    def fillsomecell(self):
        filled = list()
        # make allowed array
        allowed = self.make_allowedmap()
        for r in range(self.size):
            for c in range(self.size):
                if len(allowed[r*self.size + c]) == 1 :
                    continue
                #print("allowed at ", allowed[r*self.size + c])
                for i in allowed[r*self.size+c]:
                    s = SudokuSolver(self)
                    s.put(r,c,i)
                    a = copy.deepcopy(allowed)
                    a[r*self.size+c].discard(i)
                    #print(s,a,"\n",i)
                    if s.tighten(a) :
                        filled.append(s)
                        # print("tighten by put ",r,c,i)
                        # print(s)
                    # else:
                    #     print("false by ",r,c,i)
        return filled
    #
    # def filled(self):
    #     return sum([1 if n != 0 else 0 for n in self.cells])
    #
    # def isfilledout(self):
    #     for n in self.cells:
    #         if n == 0 :
    #             return False
    #     return True
    #
    def solve(self):
        frontier = list()
        frontier.append(self)
        done = set()
        done.add(self)
        counter = 0
        while bool(frontier) :
            sdok = frontier.pop(0)
            counter += 1    
            if counter % 1000 == 0:
                print(sdok,counter,len(frontier),len(done))
            # if sdok in done:
            #     continue
            done.add(sdok)
            if sdok.issolved() :
                return sdok
            for each in sdok.fillsomecell():
                if each not in done:
                    frontier.append(each)
        return None
    
if __name__ == '__main__':
    #sudoku = SudokuSolver('000503000260080051300000008070000020000702000508030107001604500050020040002050700')
    #sudoku = SudokuSolver('000310008006080000090600100509000000740090052000000409007004020000020600400069000')
    #sudoku = SudokuSolver('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    #sudoku = SudokuSolver('615830049304291076000005081007000100530024000000370004803000905170900400000002003')
    #sudoku = SudokuSolver('900000000700008040010000079000974000301080000002010000000400800056000300000005001')
    #sudoku = SudokuSolver('400080100000209000000730000020001009005000070090000050010500400600300000004007603')
    #sudoku = SudokuSolver('020000010004000800060010040700209005003000400050000020006801200800050004500030006')
    #sudoku = SudokuSolver('001503900040000080002000500010060050400000003000201000900080006500406009006000300')
    #sudoku = SudokuSolver('080100000000070016610800000004000702000906000905000400000001028450090000000003040')
    #sudoku = SudokuSolver('001040600000906000300000002040060050900302004030070060700000008000701000004020500')
    #sudoku = SudokuSolver('000007002001500790090000004000000009010004360005080000300400000000000200060003170')
    sudoku = SudokuSolver('001000000807000000000054003000610000000700000080000004000000010000200706050003000')
    
    print(sudoku)
    dt = datetime.datetime.now()
    solved = sudoku.solve()
    delta = datetime.datetime.now() - dt
    print(delta.seconds*1000+ delta.microseconds/1000)
    print(solved)
