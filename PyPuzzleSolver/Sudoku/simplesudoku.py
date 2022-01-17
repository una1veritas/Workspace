'''
Created on 2021/10/31

@author: Sin Shimozono
'''
import math

class SudokuSolver():
    
    def __init__(self, size=9, grid=None):
        if int(math.sqrt(size))**2 != size:
            raise RuntimeError('Invalid size.')
        self.size = int(size)
        self.sizefactor = int(math.sqrt(self.size))
        self.boxes = [0 for i in range(self.size**2)]
        if isinstance(grid, (str,list)):
            for i in range(len(self.boxes)):
                self.boxes[i] = int(grid[i])
    
    def index(self, row, col):
        return row*self.size+col
    
    def at(self, row, col):
        return self.boxes[self.index(row,col)]

    def put(self, row, col, num):
        self.boxes[self.index(row,col)] = num

    def __str__(self):
        tmp = ''
        for r in range(self.size):
            for c in range(self.size):
                tmp += str(self.at(r, c) if self.at(r, c) != 0 else ' ')
                if c % self.sizefactor == self.sizefactor - 1:
                    tmp += '|'
                else:
                    tmp += ' '
            tmp += '\n'
            if r % self.sizefactor == self.sizefactor - 1:
                tmp += '-----+-----+-----+\n'
        return tmp
    
    def affectboxes(self,row,col):
        if 0 <= row < self.size and 0 <= col < self.size:
            baserow = (row // self.sizefactor)*self.sizefactor
            basecol = (col // self.sizefactor)*self.sizefactor
            for r in range(baserow, baserow+self.sizefactor):
                for c in range(basecol, basecol+self.sizefactor):
                    if r == row and c == col : continue
                    yield (r, c)
            for c in range(0,basecol,1):
                yield (row, c)
            for c in range(basecol+self.sizefactor,self.size,1):
                yield (row, c)
            for r in range(0,baserow,1):
                yield (r, col)      
            for r in range(baserow+self.sizefactor,self.size,1):
                yield (r, col)      

    def allowednum(self, row, col):
        if self.at(row, col) != 0:
            return {self.at(row, col)}
        cand = set([1,2,3,4,5,6,7,8,9])
        for (r,c) in self.affectboxes(row,col):
            cand.discard(self.at(r,c))
        return cand
    
    def refine(self):
        while True:
            print(sudoku)
            fixable = list()
            for (r,c) in [(r,c) for r in range(9) for c in range(9)]:
                if self.at(r,c) != 0 :
                    continue
                cand = self.allowednum(r,c)
                if len(cand) == 1:
                    fixnum = cand.pop()
                    fixable.append((r,c,fixnum))
            if len(fixable) == 0 :
                break
            for (r,c,num) in fixable:
                self.put(r,c,num)
        return 
            
    def solved(self):
        for each in self.boxes:
            if each != 0 :
                return False
        return True

sudoku = SudokuSolver(9, '003020600900305001001806400008102900700000008006708200002609500800203009005010300')
sudoku.refine()