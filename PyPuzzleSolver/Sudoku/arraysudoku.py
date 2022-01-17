#
#
import math
from array import array
import datetime

class SudokuSolver():
    SIZE_FACTORS = {4:2,9:3,16:4}
    
    def __init__(self, numbers):
        self.size = int(math.sqrt(len(numbers)))
        if self.size not in self.SIZE_FACTORS:
            raise ValueError('illegal size specified: factor = {}, size = {}, number list length = {}'.format(self.factor,self.size,len(numbers)))
        if isinstance(numbers, (str, list)) :
            self.cells = array('B',[int(d) for d in numbers])
        elif isinstance(numbers, (array)) :
            self.cells = array('B',numbers)
        else:
            raise TypeError('illegal arguments for constructor.')
    
    def __str__(self):
        factor = self.SIZE_FACTORS[self.size]
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
        for val in self.cells:
            hashval = (hashval<<3) ^ val
        return hashval
    
    def __eq__(self, another):
        if isinstance(another, SudokuSolver) :
            return self.cells == another.cells
        return False 
    
    def at(self, row, col):
        return self.cells[self.index(row,col)]

    def put(self, row, col, num):
        self.cells[self.index(row,col)] = num
    
    def index(self, row, col):
        return row*self.size+col
    
    def issolved(self):
        rownums = set()
        colnums = set()
        blocknums = set()
        
        for r in range(self.size) :
            rownums.clear()
            colnums.clear()
            for c in range(self.size) :
                num = self.at(r,c)
                if 0 < num <= self.size and num not in rownums :
                    rownums.add(num)
                else:
                    return False
                num = self.at(c,r)
                if 0 < num <= self.size and num not in colnums :
                    colnums.add(num)
                else:
                    return False
        for row in range(0,self.size,self.SIZE_FACTORS[self.size]):
            for col in range(0,self.size,self.SIZE_FACTORS[self.size]):
                blocknums.clear()
                for r in range(row, row+self.SIZE_FACTORS[self.size]):
                    for c in range(col,col+self.SIZE_FACTORS[self.size]):
                        num = self.at(r,c)
                        if 0 < num <= self.size and num not in blocknums:
                            blocknums.add(num)
                        else:
                            return False
        return True   
    
    def affectcells(self,row,col):
        factor = self.SIZE_FACTORS[self.size]
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        baserow = (row // factor)*factor
        basecol = (col // factor)*factor
        for r in range(baserow, baserow+factor):
            for c in range(basecol, basecol+factor):
                if r == row and c == col : continue
                yield (r, c)
        for c in range(0,basecol,1):
            yield (row, c)
        for c in range(basecol+factor,self.size,1):
            yield (row, c)
        for r in range(0,baserow,1):
            yield (r, col)      
        for r in range(baserow+factor,self.size,1):
            yield (r, col)

    def allowednumbers(self, row, col):
        if self.at(row, col) != 0:
            return set()
        cands = set([i for i in range(1,self.size+1)])
        for (r,c) in self.affectcells(row,col):
            cands.discard(self.at(r,c))
        return cands
    
    def filluniquepossibility(self):
        tobefixed = set()
        for r in range(self.size): 
            for c in range(self.size):
                if self.at(r,c) == 0 :
                    tobefixed.add( (r,c) )
        while bool(tobefixed):
            #print(sudoku)
            (r,c) = tobefixed.pop()
            cand = self.allowednumbers(r,c)
            #print(cand)
            if len(cand) == 0:
                return False
            elif len(cand) == 1:
                num = cand.pop()
                self.put(r,c,num)
                for row, col in self.affectcells(r, c):
                    # if row == r and col == c:
                    #     continue
                    if self.at(row,col) == 0:
                        tobefixed.add((row,col))
            #print(len(tobefixed),end=",")
            #self.checkAll()
        #print()
        return True

    def possibilitymap(self):
        possmap = dict()
        for r in range(self.size): 
            for c in range(self.size):
                possmap[(r,c)] = set([1,2,3,4,5,6,7,8,9])
        #row rules
        for r in range(self.size): 
            fixednums = set([self.at(r,c) for c in range(self.size) if self.at(r,c) != 0])
            for c in range(self.size): 
                if self.at(r,c) != 0 :
                    possmap[(r,c)] = set([self.at(r,c)])
                else:
                    possmap[(r,c)] -= fixednums
        #column rules
        for c in range(self.size): 
            fixednums = set([self.at(r,c) for r in range(self.size) if self.at(r,c) != 0])
            for r in range(self.size): 
                if self.at(r,c) == 0 :
                    possmap[(r,c)] -= fixednums
        #block rules 
        szfactor = self.SIZE_FACTORS[self.size]
        for br in range(szfactor): 
            for bc in range(szfactor): 
                fixednums = set()
                for r in range(br*szfactor,(br+1)*szfactor):
                    for c in range(bc*szfactor,(bc+1)*szfactor):
                        if self.at(r,c) != 0 :
                            fixednums.add(self.at(r,c))
                for r in range(br*szfactor,(br+1)*szfactor):
                    for c in range(bc*szfactor,(bc+1)*szfactor):
                        if self.at(r,c) == 0 :
                            possmap[(r,c)] -= fixednums
        return possmap
    
    def updatepossibilitymap(self, possmap, row, col, num):
        possmap[(row,col)] = set([num])
        for (r,c) in self.affectcells(row,col):
            if self.at(r,c) == 0 :
                possmap[(r,c)] -= set([num])
        
    def trytofill(self):
        updated = True
        while updated :
            # next trial
            updated = False 
            # fix unique possible numbers
            if not self.filluniquepossibility() :
                return False
            #
            # for (r,c) in possmap:
            #     if self.at(r,c) == 0 and len(possmap[(r,c)]) == 1 :
            #         elem = list(possmap[(r,c)])[0]
            #         self.put(r,c,elem)
            #         self.updatepossibilitymap(possmap, r, c, elem)
            #         updated = True
            # if updated: continue
            # two possible number cells
            
                    
        # for k in possmap:
        #     print(k,possmap[k])
        # print(self, self.filled())
        # exit()
        return True
    
    def guessed(self):
        filled = list()
        for r in range(self.size):
            for c in range(self.size):
                for i in self.allowednumbers(r,c):
                    s = SudokuSolver(self.cells)
                    s.put(r,c,i)
                    # print(r,c,i)
                    filled.append(s)
        return filled

    def filled(self):
        return sum([1 if n != 0 else 0 for n in self.cells])
    
    def isfilledout(self):
        for n in self.cells:
            if n == 0 :
                return False
        return True
    
    # def solve(self):
    #     frontier = list()
    #     frontier.append(self)
    #     done = set()
    #     counter = 0
    #     while len(frontier) > 0 :
    #         sd = frontier.pop(0)
    #         if not sd.tighten() :
    #             continue
    #         if sd not in done:
    #             done.add(sd)
    #         else:
    #             continue
    #         counter += 1    
    #         if counter % 1000 == 0:
    #             print(sd,counter,len(frontier), len(done))
    #         if sd.isfilledout() :
    #             return sd
    #         for nx in sd.fillsomecell():
    #             if not nx in done:
    #                 frontier.append(nx) 
    #
    #         #frontier.extend(nextgen)
    #     return None
    
    def solve(self):
        self.trytofill()
        frontier = [self]
        nextgen = set()
        counter = 0
        while len(frontier) > 0 :
            sd = frontier.pop(0)
            if sd.isfilledout() :
                return sd
            counter += 1
            if counter % 100 == 0:
                print("{}counter={}, frontier={}, nextgen={}\n".format(sd,counter,len(frontier), len(nextgen)))
            for nx in sd.guessed():
                #nextgen.add(nx) 
                if nx.trytofill() :
                    frontier.append(nx) 
            # if not bool(frontier) :
            #     frontier = nextgen
            #     nextgen = set()
            #     #print("{}counter={}, frontier={}, nextgen={}\n".format(sd,counter,len(frontier), len(nextgen)))
        return None
    
if __name__ == '__main__':
    #sudoku = SudokuSolver('000503000260080051300000008070000020000702000508030107001604500050020040002050700')
    #sudoku = SudokuSolver('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    #sudoku = SudokuSolver('615830049304291076000005081007000100530024000000370004803000905170900400000002003')
    #sudoku = SudokuSolver('900000000700008040010000079000974000301080000002010000000400800056000300000005001')
    #sudoku = SudokuSolver('400080100000209000000730000020001009005000070090000050010500400600300000004007603')
    #sudoku = SudokuSolver('020000010004000800060010040700209005003000400050000020006801200800050004500030006')
    #sudoku = SudokuSolver('000310008006080000090600100509000000740090052000000409007004020000020600400069000')
    #sudoku = SudokuSolver('080100000000070016610800000004000702000906000905000400000001028450090000000003040')
    #sudoku = SudokuSolver('001503900040000080002000500010060050400000003000201000900080006500406009006000300')
    sudoku = SudokuSolver('000007002001500790090000004000000009010004360005080000300400000000000200060003170')
    #sudoku = SudokuSolver('001040600000906000300000002040060050900302004030070060700000008000701000004020500')
    #sudoku = SudokuSolver('001000000807000000000054003000610000000700000080000004000000010000200706050003000')
    
    print(sudoku, sudoku.filled())
    dt = datetime.datetime.now()
    solved = sudoku.solve()
    delta = datetime.datetime.now() - dt
    print(solved, "Solved." if solved.issolved() else "Not Solved!")
    print(delta.seconds*1000+ delta.microseconds/1000)
