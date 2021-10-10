'''
Created on 2021/10/10

@author: Sin Shimozono
'''

if __name__ == '__main__':
    pass
else:
    exit()

class FillinPuzzle:
    
    def __init__(self, places, values, defval = None, assigns = None):
        self.map = dict()
        self.values = set(values)
        if defval == None:
            self.defaultvalue = list(values)[0]
        else:
            self.defaultvalue = defval
        for key in places:
            self.map[key] = self.defaultvalue
        if assigns == None:
            return
        if isinstance(assigns, dict) :
            for key in assigns :
                self.map[key] = assigns[key]
        elif isinstance(assigns, str) :
            if len(assigns) != 81 :
                return
            for i in range(81):
                self.map[(i//9, i % 9)] = int(assigns[i])
    
    def __str__(self):
        hborder = '+-----+-----+-----+'
        mapstr = hborder + '\n'
        for r in range(9):
            mapstr += '|'
            for c in range(9):
                mapstr += str(self.map[(r,c)])
                if c % 3 == 2 :
                    mapstr += '|'
                else:
                    mapstr += ' '
            mapstr += '\n'
            if r % 3 == 2:
                mapstr += hborder
                if r != 8 :
                    mapstr += '\n'
        return self.__class__.__name__ + '(\n' + mapstr + ', ' + str(self.values) +') '

    def row(self, num):
        r = num
        for c in range(0, 9):
            yield (r,c)

    def column(self, num):
        c = num
        for r in range(0, 9):
            yield (r,c)
    
    def block(self, row, col):
        for r in range(row*3, row*3 + 3):
            for c in range(col*3, col*3+3):
                yield (r,c)

    def emptycells(self):
        for key in sorted(self.map.keys()):
            if self.map[key] == 0 :
                yield key
    
sudoku = list()
for r in range(9):
    for c in range(9):
        sudoku.append( (r,c) ) 
prob = "037080050409650000000400001672508490000709006804203517046000108300070000000800340"
puzzle = FillinPuzzle(sudoku, [i for i in range(10)], 0, prob)

print(puzzle)
print('row')
for key in puzzle.block(1,2):
    print(key, puzzle.map[key])
print('column')
for key in puzzle.column(2):
    print(key, puzzle.map[key])
print('block')
for key in puzzle.row(3):
    print(key, puzzle.map[key])
print('empty cells')
iter = puzzle.emptycells()
for key in iter:
    print(key, puzzle.map[key])
