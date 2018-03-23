'''
Created on 2018/03/22

@author: sin
'''
import sys

class SequencePattern:
    '''
    classdocs
    '''
    patternlist = []


    def __init__(self, patternstr):
        '''
        Constructor
        '''
        self.parsestring(patternstr)
        
    def __str__(self):
        return str(self.patternlist)
        
    def parsestring(self, patstr):
        patstr = patstr.lstrip().rstrip()
        self.patternlist.clear()
        ix = 0
        while ix < len(patstr) :
            leftix = patstr.find('[', ix)
            if ix == -1 : break
            rightix = patstr.find(']', ix + 1)
            clausestr = patstr[leftix:rightix+1]
            clause = []
            for eqstr in clausestr[1:-1].split(','):
                clause.append(eqstr.strip())
            self.patternlist.append(clause)
            ix = rightix + 1
        return
    
    def length(self):
        return len(self.patternlist)

    def match_clause(self, patclause, seqtuple, assigns):
        result = True
        subs = { }
        for lit in range(len(patclause)):
            print(patclause[lit], seqtuple[lit] )
            if patclause[lit] == '*' : 
                continue
            elif patclause[lit] in assigns:
                if assigns[patclause[lit]] != seqtuple[lit] :
                    result = False
                    subs.clear()
                    break
            elif patclause[lit] in subs:
                if subs[patclause[lit]] != seqtuple[lit] :
                    result = False
                    subs.clear()
                    break
            else:
                try:
                    if eval(patclause[lit]) != seqtuple[lit]:
                        result = False
                        subs.clear()
                        break
                except ( NameError ):
                    subs[patclause[lit]] = seqtuple[lit]
        return (result, subs)

    def eval_clause(self, eqnlist, assigns):
        result = True
        for eqnstr in eqnlist:
            try:
                tmpres = eval(eqnstr,{},assigns)
            except  (ValueError, SyntaxError):
                print('eva;_exprs value/syntax error: ', eqnlist, assigns)
                tmpres = False
            if isinstance(tmpres, bool):
                result = result and tmpres
            else:
                print('eva;_exprs error: ', eqnlist, assigns)
                result = False
            if not result : break
        return result, { }
    
    def match(self, tsseq):
        assignments = { }
        flag = True
        tupleix = 0
        clauseix = 0
        while ( clauseix < self.length() ):
            if not (tupleix < len(tsseq)) :
                return False
            if self.patternlist[clauseix][0] == '?' :
                eqnlist = self.patternlist[clauseix][1:]
                res, subs = self.eval_clause(eqnlist, assignments)
                clauseix = clauseix + 1
            else:
                res, subs = self.match_clause(self.patternlist[clauseix], tsseq[tupleix], assignments)
                clauseix = clauseix + 1
                tupleix = tupleix + 1
            if not res:
                return False
            for k in subs:
                assignments[k] = subs[k]
        print(assignments)
        return flag


if __name__ == '__main__':
    print(sys.argv)
    seqpatt = SequencePattern(sys.argv[1])
    print(seqpatt)
    tsseq = eval(sys.argv[2])
    print(tsseq)
    print(seqpatt.match(tsseq))
    