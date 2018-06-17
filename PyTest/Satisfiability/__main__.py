import sys
import random
import ast

class BooleanFormula():
    def __init__(self,f):
        self.formula = f
        boolvars = set()
        for node in ast.walk(ast.parse(f)):
            if isinstance(node, ast.Name) :
                boolvars.add(node.id)
        self.variables = tuple(sorted(list(boolvars)))

    def evaluate(self, assign):
        return eval(self.formula, assign)

#引数には python の論理式として書かれた論理関数
f = BooleanFormula(sys.argv[1])
print('BooleanFormula'+str(f.variables)+' = \n   '+str(f.formula))

bassign = { }
for bvar in f.variables:
    bassign[bvar] = (random.getrandbits(1) == 1)

print('assignment = ' + str(bassign))
if f.evaluate(bassign) :
    print(True)
else:
    print(False)
