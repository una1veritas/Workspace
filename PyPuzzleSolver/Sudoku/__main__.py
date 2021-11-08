#
#
import math

class Sudoku():
    def __init__(self, numbers):
        self.size = int(math.sqrt(len(numbers)))
        self.factor = int(math.sqrt(self.size))
        if self.factor**2 != self.size or self.size**2 != len(numbers):
            raise ValueError('illegal size specified: factor = {}, size = {}, number list length = {}'.format(self.factor,self.size,len(numbers)))
        if isinstance(numbers, (str, list)) :
            self.cells = [int(d) for d in numbers]
        else:
            raise TypeError('illegal arguments for constructor.')
    
    def __str__(self):
        tmp = ''
        for r in range(self.size):
            for c in range(self.size):
                tmp += str(self.at(r, c) if self.at(r, c) != 0 else ' ')
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
        for val in self.cells:
            hashval = (hashval*(self.factor+1)) ^ val
        return hashval
    
    def __eq__(self, another):
        if isinstance(another, Sudoku) :
            return self.cells == another.cells
        return False 
    
    def at(self, row, col):
        return self.cells[self.index(row,col)]

    def put(self, row, col, num):
        self.cells[self.index(row,col)] = num
    
    def index(self, row, col):
        return row*self.size+col
    
    def issolved(self):
        return 0 not in [v for v in self.cells]
    
    def check(self,row,col):
        rownums = set()
        colnums = set()
        blocknums = set()
        for c in range(0,9):
            num = self.at(row,c)
            if num > 0 and num in rownums :
                return False
            rownums.add(num)
        for r in range(0,9):
            num = self.at(r,col)
            if num > 0 and num in colnums:
                return False
            colnums.add(num)
        for r in range((row//3)*3,(row//3)*3+3):
            for c in range((col//3)*3,(col//3)*3):
                num = self.at(r,c)
                if num > 0 and num in blocknums:
                    return False
                blocknums.add(num)
        return True   
    
    # def checkAll(self):
    #     for r in range(0,9):
    #         for c in range(0,9):
    #             if not self.check(r,c):
    #                 print('\n'+str(self))
    #                 raise RuntimeError('I have bad feeling at {},{}.'.format(r,c))
    
    def affectcells(self,row,col):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return
        baserow = (row // self.factor)*self.factor
        basecol = (col // self.factor)*self.factor
        for r in range(baserow, baserow+self.factor):
            for c in range(basecol, basecol+self.factor):
                if r == row and c == col : continue
                yield (r, c)
        for c in range(0,basecol,1):
            yield (row, c)
        for c in range(basecol+self.factor,self.size,1):
            yield (row, c)
        for r in range(0,baserow,1):
            yield (r, col)      
        for r in range(baserow+self.factor,self.size,1):
            yield (r, col)

    def allowednumbers(self, row, col):
        if self.at(row, col) != 0:
            return set()
        cands = set([1,2,3,4,5,6,7,8,9])
        for (r,c) in self.affectcells(row,col):
            cands.discard(self.at(r,c))
        return cands
    
    def tighten(self):
        counter = 0
        while True:
            #print(sudoku)
            fix = None
            for (r,c) in [(r,c) for r in range(9) for c in range(9)]:
                if self.at(r,c) == 0 :
                    cand = self.allowednumbers(r,c)
                    if len(cand) == 0:
                        return False
                    elif len(cand) == 1:
                        fix = (r,c,cand.pop())
                        break
            if fix == None:
                break
            counter += 1
            (r,c,num) = fix
            self.put(r,c,num)
            # print("({},{}) <- {}".format(r,c,num))
            #self.checkAll()
            #print()
        return True
    
    def fillsomecell(self):
        filled = list()
        for r in range(self.size):
            for c in range(self.size):
                for i in self.allowednumbers(r,c):
                    s = Sudoku(self.cells)
                    s.put(r,c,i)
                    # print(r,c,i)
                    filled.append(s)
        return filled

    def filllevel(self):
        return sum([1 if n != 0 else 0 for n in self.cells])
    
    def solve(self):
        solved = None
        if not self.tighten() : return solved
        frontier = dict()
        level = self.filllevel()
        frontier[level] = set()
        frontier[level].add(self)
        while len(frontier) > 0 :
            if len(frontier[level]) == 0 :
                frontier.pop(level)
                level += 1
                continue
            sd = frontier[level].pop()
            #sd.checkAll()
            if len(frontier[level]) < 100 or (len(frontier[level]) % 100) == 0 :
                print(level, len(frontier[level]))
                print(sd)
            if sd.issolved():
                solved = sd
                break
            nextgen = sd.fillsomecell()
            for next in nextgen: 
                if next.tighten():
                    nextlevel = next.filllevel()
                    if nextlevel not in frontier:
                        frontier[nextlevel] = set()
                    frontier[nextlevel].add(next)
            #frontier.extend(nextgen)
        return solved
    
if __name__ == '__main__':
    #sudoku = Sudoku('003020600900305001001806400008102900700000008006708200002609500800203009005010300')
    #sudoku = Sudoku('615830049304291076000005081007000100530024000000370004803000905170900400000002003')
    #sudoku = Sudoku('900000000700008040010000079000974000301080000002010000000400800056000300000005001')
    #sudoku = Sudoku('400080100000209000000730000020001009005000070090000050010500400600300000004007603')
    sudoku = Sudoku('020000010004000800060010040700209005003000400050000020006801200800050003500030006')
    #sudoku = Sudoku('000007002001500790090000004000000009010004360005080000300400000000000200060003170')
    #print(sudoku)
    solved = sudoku.solve()    
    print(solved)
