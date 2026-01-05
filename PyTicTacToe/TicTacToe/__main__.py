'''
Created on 2026/01/05

@author: sin
'''
import copy

class TicTacToe:
    def __init__(self):
        self.board = [' ']*9
        self.player = 0
    
    def __str__(self):
        result = ''
        for rix in range(5) :
            for cix in range(5) :
                if (rix % 2) == 1 and (cix % 2) == 0 :
                    result += '-'
                elif (rix % 2) == 1 and (cix % 2) == 1 :
                    result += '+'
                elif (cix % 2) == 1 :
                    result += '|'
                elif (cix % 2) == 0 :
                    result += self.board[(rix//2)*3 +(cix // 2)]
            result += '\r\n'
        return result
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, another):
        if isinstance(another, TicTacToe) :
            return another.board == self.board
        return False
    
    def __hash__(self):
        value = 0
        for r in range(3) :
            for c in range(3) :
                value <<= 2
                value |= 0 if self.cell(r,c) == ' ' else 1 if self.cell(r,c) == 'O' else 2
        return value
    
    def place(self, r, c):
        if self.board[r*3+c] != ' ' :
            raise ValueError(f'Cell {r}, {c} is not empty!')
        self.board[r*3+c] = 'X' if self.player == 0 else 'O'
        self.player = (self.player + 1) % 2
    
    def cell(self, r, c):
        return self.board[r*3+c]
    
    def winlosedraw(self):
        for rix in range(3) :
            if all([self.cell(rix,c) == 'O' for c in range(3)]) :
                return True
            if all([self.cell(rix, c) == 'X' for c in range(3)]) :
                return True
        for cix in range(3) :
            if all([self.cell(r, cix) == 'O' for r in range(3)]) :
                return True
            if all([self.cell(r,cix) == 'X' for r in range(3)]) :
                return True
        if all([self.cell(r, r) == 'O' for r in range(3)]) :
            return True
        if all([self.cell(r, r) == 'X' for r in range(3)]) :
            return True
        if all([self.cell(r,2-r) == 'O' for r in range(3)]) :
            return True
        if all([self.cell(r, 2-r) == 'X' for r in range(3)]) :
            return True
        if all([self.cell(r,c) != ' ' for r in range(3) for c in range(3)]) :
            return True
        return False

def depthfirstsearch(board, nodes, edges):
    #print(board)
    for r in range(3):
        for c in range(3):
            if board.cell(r,c) == ' ' :
                newboard = copy.deepcopy(board)
                newboard.place(r,c)
                nodes.add(newboard)
                edges.append({board, newboard})
                if not newboard.winlosedraw() :
                    depthfirstsearch(newboard, nodes, edges)
    return
    
if __name__ == '__main__':
    ttt = TicTacToe()
    print(ttt)
    nodes = set()
    edges = list()
    depthfirstsearch(ttt, nodes, edges)
    print(len(nodes))
    for each in list(nodes) :
        if each.winlosedraw() :
            print(each)