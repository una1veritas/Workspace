#
#
import math, copy
from array import array

class Sudoku():
    def __init__(self, arg):
        if isinstance(arg,(str,list)):
            numbers = arg
            if len(numbers) not in [16,81,256]:
                raise ValueError('argument array has illegal size = {}'.format(len(numbers)))            
            self.size = int(math.sqrt(len(numbers)))
            self.array = array('H',[self.bits(0) for i in range(len(numbers))])
            for i in range(len(numbers)):
                num = int(numbers[i])
                if 0 <= num <= self.size : 
                    self.put(self.row(i),self.col(i),num)
                    #print(self.row(i),self.col(i),num)
                else:
                    raise ValueError('array element is illegal value = {}'.format(numbers[i]))
        elif isinstance(arg,Sudoku):
            self.size = arg.size
            self.array = copy.copy(arg.array)
            
        
    def bits(self, num):
        if num == 0 : 
            return (1<<self.size) - 1
        if num == -1 :
            return 0
        return 1<<(num-1)
    
    def debits(self, bval):
        if self.size == 9 :
            if bval == 0 : return -1
            if bval == (1<<self.size) - 1 :return 0
            val = 1
            while bval != 0 and (bval & 1) == 0:
                val += 1
                bval >>= 1
            if (bval>>1) == 0 :
                return val
        elif self.size == 16:
            pass
        elif self.size == 4:
            pass
        return 0
    
    def _popcount(self,val):
        pcntbl = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
        count = 0
        while val != 0 :
            count += pcntbl[val & 0x0f]
            val >>= 4
        return count
    
    def row(self,i):
        return i // self.size
    
    def col(self,i):
        return i % self.size
    
    def index(self,row,col):
        return row*self.size + col

    def put(self,row,col,num):
        if num == 0 :
            return True
        #print(bin(self.bits_at(row, col)),num)
        if self.bits_at(row, col) & self.bits(num) == 0 :
            raise RuntimeError('Illegal placement of number {} at {},{}'.format(num,row,col))
        updates = list()
        updates.append((row,col,num))
        while bool(updates) :
            row,col,num = updates.pop(0)
            self.array[self.index(row,col)] = self.bits(num)
            for r, c in self.affectedarea(row,col):
                if r == row and c == col :
                    continue
                bits = self.array[self.index(r,c)]
                newbits = bits & (self.bits(0) ^ self.bits(num))
                self.array[self.index(r,c)] = newbits
                newpcnt = self._popcount(newbits)
                if newpcnt == 0 :
                    return False
                if newbits != bits and newpcnt == 1 :
                    updates.append((r,c,self.debits(newbits)))
        return True
        
    def bits_at(self,row,col):
        return self.array[row*self.size+col]
    
    def at(self,row,col):
        return self.debits(self.bits_at(row,col))
    
    def signature(self):
        return ''.join([str(self.debits(ea)) for ea in self.array])
    
    def __str__(self):
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
        factor = int(math.sqrt(self.size))
        for val in self.array:
            hashval = (hashval<<factor) ^ val
        return hashval
    
    def __eq__(self, another):
        if isinstance(another, Sudoku) :
            return self.array == another.array
        return False 
    
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
        for bits in self.array:
            if self._popcount(bits) != 1 :
                return False
        return True
    
    def affectedarea(self,row,col):
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
    
    def allowed(self, row, col):
        b = self.bits_at(row, col)
        if self._popcount(b) == 1 :
            return
        num = 1
        while b != 0:
            if b & 1 == 1 :
                yield num
            b >>= 1
            num += 1
    #
    # def tighten(self):
    #     while True:
    #         #print(sudoku)
    #         fix = None
    #         for r in range(9):
    #             for c in range(9):
    #                 if self.at(r,c) != 0 : 
    #                     continue
    #                 cand = self.allowednumbers(r,c)
    #                 #print(cand)
    #                 if len(cand) == 0:
    #                     return False
    #                 elif len(cand) == 1:
    #                     fix = (r,c,cand.pop())
    #                     break
    #         if fix == None :
    #             return True
    #         (r,c,num) = fix
    #         self.put(r,c,num)
    #         #print("({},{}) <- {}".format(r,c,num))
    #         #self.checkAll()
    #         #print()
    #     return True
    #
    def fillsomecell(self):
        filled = list()
        for r in range(self.size):
            for c in range(self.size):
                for i in self.allowed(r,c):
                    s = Sudoku(self)
                    if s.put(r,c,i):
                        # print(r,c,i)
                        filled.append(s)
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
            if counter % 500 == 0:
                print(sdok,counter,len(frontier),len(done))
            # if sdok in done:
            #     continue
            done.add(sdok)
            if sdok.issolved() :
                return sdok
            for each in sdok.fillsomecell():
                if each not in done:
                    frontier.append(each)
                    done.add(each)
        return None
    
if __name__ == '__main__':
    #sudoku = Sudoku('000310008006080000090600100509000000740090052000000409007004020000020600400069000')
    #sudoku = Sudoku('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    #sudoku = Sudoku('615830049304291076000005081007000100530024000000370004803000905170900400000002003')
    #sudoku = Sudoku('900000000700008040010000079000974000301080000002010000000400800056000300000005001')
    #sudoku = Sudoku('400080100000209000000730000020001009005000070090000050010500400600300000004007603')
    #sudoku = Sudoku('020000010004000800060010040700209005003000400050000020006801200800050004500030006')
    #sudoku = Sudoku('001503900040000080002000500010060050400000003000201000900080006500406009006000300')
    #sudoku = Sudoku('080100000000070016610800000004000702000906000905000400000001028450090000000003040')
    sudoku = Sudoku('001040600000906000300000002040060050900302004030070060700000008000701000004020500')
    #sudoku = Sudoku('000007002001500790090000004000000009010004360005080000300400000000000200060003170')
    print(sudoku)
    print(sudoku.signature())
    solved = sudoku.solve()
    print(solved, solved.issolved())
