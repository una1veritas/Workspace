'''
Created on 2018/03/22

@author: sin
'''
import sys

class SequencePattern:
    '''
    classdocs
    '''
    pattSeq = []


    def __init__(self, patternstr):
        '''
        Constructor
        '''
        self.parseStr(patternstr)
        
    def __str__(self):
        return str(self.pattSeq)
        
    def parseStr(self, patstr):
        patstr = patstr.lstrip().rstrip()
        self.pattSeq.clear()
        ix = 0
        while ix < len(patstr) :
            leftix = patstr.find('[', ix)
            if ix == -1 : break
            rightix = patstr.find(']', ix + 1)
            clausestr = patstr[leftix:rightix+1]
            clause = []
            for eqstr in clausestr[1:-1].split(','):
                clause.append(eqstr.strip())
            self.pattSeq.append(clause)
            ix = rightix + 1
        return
    
    def length(self):
        return len(self.pattSeq)

    def ispredicate(self, ith):
        return self.pattSeq[ith][0] == '?'

    def clauseMatch(self, patclause, seqtuple, assigns):
        result = True
        subs = { }
        for lit in range(len(patclause)):
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

    def clauseEval(self, ith, maps):
        result = True
        predlist = self.pattSeq[ith][1:]
        vardict = {}
        for varmap in maps:
            for k in varmap:
                vardict[k] = varmap[k]
        for predstr in predlist:
            try:
                tmpres = eval(predstr,{},vardict)
            except  (ValueError, SyntaxError):
                print('eva;_exprs value/syntax error: ', predstr, vardict)
                tmpres = False
            if isinstance(tmpres, bool):
                result = result and tmpres
            else:
                print('eva;_exprs error: ', predstr, vardict)
                result = False
            if not result : break
        return result, { }
    
    def match(self, seq, pos):
        maps = []
        ith = 0
        while ( ith < self.length() ):
            if self.ispredicate(ith) :
                res, subs = self.clauseEval(ith, maps)
                # maps is read-only
                ith = ith + 1
            else:
                if not (pos < len(seq) ) :
                    return False
                res, subs = self.clauseMatch(self.pattSeq[ith], seq[pos], maps)
                # the last clause must have no defined variables
                maps.append(subs)
                ith = ith + 1                
                pos = pos + 1
            if not res:
                return False
        return True


if __name__ == '__main__':
    print(sys.argv)
    seqpatt = SequencePattern(sys.argv[1])
    print(seqpatt)
    tsseq = eval(sys.argv[2])
    print(tsseq)
    for pos in range(0,len(tsseq)):
        print(seqpatt.match(tsseq,pos))
        print()
    