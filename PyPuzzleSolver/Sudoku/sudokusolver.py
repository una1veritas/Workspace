#
#
import math, copy
from array import array
import datetime

class Sudoku():
    def __init__(self, arg):
        if isinstance(arg,(str,list,tuple)):
            self.size = int(math.sqrt(len(arg)))
            self.factor = int(math.sqrt(self.size))
            if len(arg) != self.size**2 or self.size != self.factor**2 :
                raise ValueError('argument array has illegal size = {}'.format(len(arg)))            
            if isinstance(arg, (str, list, array)) :
                self.array = array('B',[int(d) if 1 <= int(d) <= self.size else 0 for d in arg])
            else:
                raise ValueError('arg is illegal value.')
            # print(self.array)
        elif isinstance(arg, SudokuSolver):
            self.size = arg.size
            self.factor = arg.factor
            self.array = copy.copy(arg.array)
    
    def __str__(self):
        tmp = ''
        
        for r in range(self.size):
            for c in range(self.size):
                if self.at(r,c) == 0 :
                    tmp += ' '
                else:
                    tmp += str(self.at(r, c))
                if c % self.factor == self.factor - 1:
                    tmp += '|'
                else:
                    tmp += ' '
            tmp += '\n'
            if r % self.factor == self.factor - 1 :
                tmp += '-----+-----+-----+\n'
        return tmp
    
    def __hash__(self):
        hashval = 0
        for val in self.array:
            hashval = (hashval<<self.factor) ^ val
        return hashval
    
    def __eq__(self, another):
        return self.array == another.array
        
    def at(self,row,col):
        return self.array[row*self.size + col]
    
    def put(self,row,col,val):
        self.array[row*self.size + col] = int(val)
    
    def isfixed(self,row,col):
        return self.at(row,col) != 0
    
    def solve(self):
        frontier = list()
        frontier.append(self)
        # done = set()
        # done.add(self)
        # counter = 0
        while bool(frontier) :
            sd = frontier.pop(0)
            solver = SudokuSolver(sd)
            print(solver)
            solver.narrow_by_nlets()
            print(solver)
            print(len(frontier))
            if sd.issolved():
                return sd        
        #     counter += 1    
        #     if counter % 1000 == 0:
        #         print(sdok,counter,len(frontier),len(done))
        #     # if sdok in done:
        #     #     continue
        #     done.add(sdok)
            frontier += sd.guess()
            
        return None
    
class SudokuSolver():
    def __init__(self, arg):
        if isinstance(arg,Sudoku):
            self.sudoku = arg
            self.hints = self.hinttable()
            #print(self.hint)
        elif isinstance(arg,SudokuSolver):
            self.sudoku = arg.sudoku
            self.hint = copy.deepcopy(arg.hint)

    @property
    def size(self):
        return self.sudoku.size
    
    @property
    def factor(self):
        return self.sudoku.factor
    
    def discard(self, row, col, num):
        try:
            self.at(row,col).remove(num)
        except ValueError:
            pass
        except AttributeError:
            pass

    def hinttable(self):
        fixedinrow = list()
        for row in range(self.size):
            fixedinrow.append(set())
            for (r,c) in self.rowcells(row,0):
                if self.sudoku.isfixed(r,c) :
                    #print(fixedinrow[-1], self.sudoku.at(r,c))
                    fixedinrow[-1].add(self.sudoku.at(r,c))
        fixedincol = list()
        for col in range(self.size):
            fixedincol.append(set())
            for (r,c) in self.columncells(0,col):
                if self.sudoku.isfixed(r,c) :
                    fixedincol[-1].add(self.sudoku.at(r,c)) 
        #print(fixedinrow)
        #print(fixedincol)
        fixedinblock = list()
        for i in range(self.size):
            fixedinblock.append(set())
            #print((i//3)*3,(i%3)*3)
            for (r,c) in self.blockcells((i//self.factor)*self.factor,(i%self.factor)*self.factor):
                if self.sudoku.isfixed(r,c) :
                    fixedinblock[-1].add(self.sudoku.at(r,c)) 
        #print(fixedinblock)
        hinttable = list()
        for r in range(self.size):
            for c in range(self.size):
                if self.sudoku.isfixed(r,c):
                    hinttable.append([self.sudoku.at(r,c)])
                else:
                    b = (r//self.factor)*self.factor+(c//self.factor)
                    hinttable.append([i for i in range(1,self.size+1)])
                    for i in fixedinrow[r].union(fixedincol[c]).union(fixedinblock[b]):
                        hinttable[-1].remove(i)
                    #print(hinttable[-1])
        #print(hinttable)
        #exit()
        return hinttable
        
    def narrow_by_fixed(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.at(row,col) is not None and self.isfixed(row,col) :
                    #print(row,col,'->',self.at(row,col), end=', ')
                    num = self.at(row,col)[0]
                    for (r,c) in self.relatecells(row, col):
                        if self.at(r,c) is None:
                            self.put(r,c,[i for i in range(1,self.size+1)])
                        if not self.isfixed(r,c):
                            self.discard(r,c,num)
                            #print(row,col,self.at(row,col),end=',')
                #print()
        return True
    
    # def index(self,row,col):
    #     return row*self.size + col

    def table_at(self,row,col):
        return self.hints[row*self.size + col]
    
    def table_put(self,row,col,val):
        self.hints[row*self.size + col] = val
    
    def table_isfixed(self, row, col):
        return len(self.table_at(row,col)) == 1
#        return isinstance(self.at(row, col), int)
    
    # def narrow_by_unique(self):
    #     uniquecells = list()
    #     for row in range(self.size):
    #         for col in range(self.size):
    #             if not self.isfixed(row,col) and len(self.at(row,col)) == 1 :
    #                 uniquecells.append( (row,col) )
    #     # print(uniquecells)
    #     while bool(uniquecells) :
    #         (row,col) = uniquecells.pop(0)
    #         num = self.at(row,col)[0]
    #         for (r,c) in self.relatecells(row, col):
    #             if self.isfixed(r,c) :
    #                 continue
    #             # print(row, col, num, '->', r, c, self.at(r,c))
    #             remained = len(self.at(r,c))
    #             self.discard(r,c,num)
    #             res = len(self.at(r,c))
    #             if remained == 1 and res == 0 :
    #                 raise RuntimeError('{}, {}'.format(r,c))
    #                 return False
    #             elif remained == 2 and res == 1 :
    #                 uniquecells.append( (r,c) )
    #     return True 
    
    def narrow_by_nlets(self):
        updated = True
        while updated:
            #print(self)
            updated = False
            for row in range(self.size):
                possrev = dict()
                for (r,c) in self.rowcells(row, 0):
                    if not self.isfixed(r,c) :
                        if tuple(self.at(r,c)) not in possrev:
                            possrev[tuple(self.at(r,c))] = list()
                        possrev[tuple(self.at(r,c))].append( (r,c) )
                #tmp = [(possrev[k], k) for k in possrev if len(k) == len(possrev[k])]
                #if bool(tmp) : print(tmp)
                for k in [t for t in possrev if len(possrev[t]) == len(t)]:
                    for (r,c) in self.row(row, 0):
                        if not self.isfixed(r,c) and (r, c) not in possrev[k]:
                            for d in k:
                                if d in self.at(r,c):
                                    self.discard(r,c,d)
                                    updated = True
    
            for col in range(self.size):
                possrev = dict()
                for (r,c) in self.column(0,col):
                    if not self.isfixed(r,c) :
                        if tuple(self.at(r,c)) not in possrev:
                            possrev[tuple(self.at(r,c))] = list()
                        possrev[tuple(self.at(r,c))].append( (r,c) )
                # tmp = [(possrev[k], k) for k in possrev if len(k) == len(possrev[k])]
                # if bool(tmp) : print(tmp)
                for k in [t for t in possrev if len(possrev[t]) == len(t)]:
                    for (r,c) in self.column(0,col):
                        if not self.isfixed(r,c) and (r, c) not in possrev[k]:
                            for d in k:
                                if d in self.at(r,c):
                                    self.discard(r,c,d)
                                    updated = True
    
            for row in range(0,self.size,self.factor):
                for col in range(0,self.size,self.factor):
                    possrev = dict()
                    for (r,c) in self.block(row,col):
                        if not self.isfixed(r,c) :
                            if tuple(self.at(r,c)) not in possrev:
                                possrev[tuple(self.at(r,c))] = list()
                            possrev[tuple(self.at(r,c))].append( (r,c) )
                    #tmp = [(possrev[k], k) for k in possrev if len(k) == len(possrev[k])]
                    #if bool(tmp) : print(tmp)
                    for k in [t for t in possrev if len(possrev[t]) == len(t)]:
                        for (r,c) in self.block(row,col):
                            if not self.isfixed(r,c) and (r, c) not in possrev[k]:
                                for d in k:
                                    #print(r,c,d,self.at(r,c))
                                    if d in self.at(r,c):
                                        self.discard(r,c,d)
                                        updated = True
        return True

    def signature(self):
        return ''.join([str(self.debits(ea)) for ea in self.array])
    
    def __str__(self):
        tmp = ''
        for r in range(self.size):
            for c in range(self.size):
                if len(self.table_at(r,c)) == 0:
                    tmp += ' '
                elif self.table_isfixed(r, c) :
                    tmp += str(self.table_at(r, c)[0])
                elif len(self.table_at(r, c)) > 1 :
                    tmp += 'x*+=-;:,. '[len(self.table_at(r, c))]

                if c % self.factor == self.factor - 1:
                    tmp += '|'
                else:
                    tmp += ' '
            tmp += '\n'
            if r % self.factor == self.factor - 1 :
                tmp += '-----+-----+-----+\n'
        return tmp
    
    def __hash__(self):
        hashval = 0
        factor = int(math.sqrt(self.size))
        for val in self.array:
            hashval = (hashval<<factor) ^ val
        return hashval
    
    def __eq__(self, another):
        if isinstance(another, SudokuSolver) :
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
        for e in self.hints:
            if len(e) > 1 :
                return False
        return True
    
    def rowcells(self,row,col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        for c in range(self.size):
            yield (row, c)

    def columncells(self,row,col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        for r in range(self.size):
            yield (r, col)
    
    def blockcells(self,row,col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        baserow = (row // self.factor)*self.factor
        basecol = (col // self.factor)*self.factor
        for r in range(baserow, baserow+self.factor):
            for c in range(basecol, basecol+self.factor):
                yield (r, c)

    def relatecells(self,row,col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        factor = self.factor
        baserow = (row // self.factor)*self.factor
        basecol = (col // self.factor)*self.factor
        for r in range(baserow, baserow+self.factor):
            for c in range(basecol, basecol+self.factor):
                if row == r and col == c :
                    continue
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
    def guess(self):
        guessed = list()
        for r in range(self.size):
            for c in range(self.size):
                if self.isfixed(r,c) :
                    continue
                for i in self.at(r,c):
                    s = SudokuSolver(self)
                    s.put(r,c,[i])
                    s.narrow_by_fixed()
                    guessed.append(s)
        return guessed
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

    
if __name__ == '__main__':
    #sudoku = SudokuSolver('000310008006080000090600100509000000740090052000000409007004020000020600400069000')
    #sudoku = SudokuSolver('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    sudoku = Sudoku('615830049304291076000005081007000100530024000000370004803000905170900400000002003')
    #sudoku = SudokuSolver('900000000700008040010000079000974000301080000002010000000400800056000300000005001')
    #sudoku = SudokuSolver('400080100000209000000730000020001009005000070090000050010500400600300000004007603')
    #sudoku = SudokuSolver('020000010004000800060010040700209005003000400050000020006801200800050004500030006')
    #sudoku = SudokuSolver('001503900040000080002000500010060050400000003000201000900080006500406009006000300')
    #sudoku = SudokuSolver('080100000000070016610800000004000702000906000905000400000001028450090000000003040')
    #sudoku = SudokuSolver('001040600000906000300000002040060050900302004030070060700000008000701000004020500')
    #sudoku = SudokuSolver('000007002001500790090000004000000009010004360005080000300400000000000200060003170')
    #sudoku = SudokuSolver('001000000807000000000054003000610000000700000080000004000000010000200706050003000')
    
    print(sudoku)
    dt = datetime.datetime.now()
    solved = sudoku.solve()
    delta = datetime.datetime.now() - dt
    print(delta.seconds*1000+ delta.microseconds/1000)
    print(solved)
    print(solved.issolved())
