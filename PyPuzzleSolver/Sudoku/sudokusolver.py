#
#
import copy
from array import array
from collections import deque
import datetime

class Sudoku():
    _SIZE_DICT = {16:(4,2), 81:(9,3), 256:(16,4)}
    
    def __init__(self, arg):
        if isinstance(arg,(str,list,tuple)):
            if len(arg) not in self._SIZE_DICT :
                raise ValueError('argument array has illegal size = {}'.format(len(arg)))
            self.size = self._SIZE_DICT[len(arg)][0]
            self.factor = self._SIZE_DICT[len(arg)][1]
            if isinstance(arg, (str, list, array)) :
                self.candidate = array('H',[self.bitrepr(int(d)) for d in arg])
                #self.narrow_by_settled()
            else:
                raise ValueError('arg is illegal value.')
            # print(self.array)
            self.history = dict()
        elif isinstance(arg, Sudoku):
            self.size = arg.size
            self.factor = arg.factor
            self.candidate = copy.copy(arg.candidate)
            self.history = copy.copy(arg.history)
        else:
            raise ValueError('arg is unexpected value.')

    def bitrepr(self, num):
        if num == 0 :
            return 0
        return 1<<(num - 1)
    
    def intval(self, bits):
        if bits == 0 or self.popcount(bits) > 1:
            return 0
        val = 1
        while (bits & 1) == 0 :
            bits >>= 1
            val += 1
        return val
    
    def as_bitset(self, bits):
        if bits == 0 :
            return
        val = 1
        while bits != 0:
            if bits & 1 == 1:
                yield val
            val += 1
            bits >>= 1
    
    def popcount(self,val):
        pcntbl = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4]
        # count = 0
        # while val != 0 :
        #     count += pcntbl[val & 0x0f]
        #     val >>= 4
        return pcntbl[val & 0x0f] + pcntbl[(val>>4) & 0x0f] + pcntbl[(val>>8) & 0x0f] + pcntbl[(val>>24) & 0x0f]

    def __str__(self):
        tmp = ''
        for r in range(self.size):
            for c in range(self.size):
                if self.isfixed(r,c) :
                    tmp += str(self.at(r, c))
                else:
                    tmp += ' '
                #     #tmp += 'x*+=-;:,. '[self.popcount(self.bitat(r,c))]
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
        return self.candidate == another.candidate
        
    def at(self,row,col):
        return self.intval(self.candidate[row*self.size + col])        

    def bitat(self,row,col):
        return self.candidate[row*self.size + col]
    
    def put(self,row,col,val):
        self.candidate[row*self.size + col] = self.bitrepr(val)
    
    def bitput(self,row,col,val):
        self.candidate[row*self.size + col] = val
    
    def isfixed(self,row,col):
        return self.popcount(self.candidate[row*self.size + col]) == 1
    
    def issolved(self):
        for r in range(self.size):
            for c in range(self.size):
                if not self.isfixed(r,c):
                    return False
        return True       
    
    def narrow_by_settled(self):
        fixed_row = list()
        for r in range(self.size):
            bitset = 0
            for c in range(self.size):
                if self.isfixed(r, c) :
                    bitset |= self.bitat(r,c)
            fixed_row.append(bitset)
        #print(fixed_row)
        fixed_col = list()
        for c in range(self.size):
            bitset = 0
            for r in range(self.size):
                if self.isfixed(r, c) :
                    bitset |= self.bitat(r,c)
            fixed_col.append(bitset)
        #print(fixed_col)
        fixed_blk = list()
        for b in range(self.size):
            bitset = 0
            baserow = (b // self.factor)*self.factor
            basecol = (b % self.factor)*self.factor
            for r in range(baserow, baserow+self.factor):
                for c in range(basecol, basecol+self.factor):
                    if self.isfixed(r, c) :
                        bitset |= self.bitat(r,c)
            fixed_blk.append(bitset)
        #print(fixed_blk)
        all1 = (1<<self.size)-1
        updated = False
        for row in range(self.size):
            for col in range(self.size):
                if not self.isfixed(row,col):
                    blk = (row//self.factor)*self.factor+(col//self.factor)
                    cands = all1^(fixed_row[row] | fixed_col[col] | fixed_blk[blk])
                    self.bitput(row,col, cands) 
                    if cands == 0 :
                        raise RuntimeError('Candidates exhausted.')
                    if self.isfixed(row,col):
                        updated = True
                        if 'settle' in self.history :
                            self.history['settle'] += 1
                        else:
                            self.history['settle'] = 1
        #print(self.candidate)
        return updated
    
    def narrow_by_candidates(self):
        # assuming no cells are exhausted.
        updated = False
        all1 = (1<<self.size)-1
        # search row groups for
        for row in range(self.size):
            candsdict = dict()
            for (r,c) in self.rowcells(row, 0):
                if self.bitat(r,c) not in candsdict:
                    candsdict[self.bitat(r,c)] = list()
                candsdict[self.bitat(r,c)].append( (r,c) )
            #print(candsdict)
            for k in [t for t in candsdict if len(candsdict[t]) == self.popcount(t)]:
                for (r,c) in self.rowcells(row, 0):
                    if not self.isfixed(r,c) and (r, c) not in candsdict[k]:
                        bits = self.bitat(r,c)
                        self.bitput(r,c,self.bitat(r,c) & (all1^k))
                        if bits != 0 and self.bitat(r,c) == 0:
                            raise RuntimeError('Exhausted candidates.')
                        if bits != self.bitat(r,c) :
                            #print('row',r,c,bits,'->',self.bitat(r,c))
                            updated = True
                            if 'candidates'+str(self.popcount(k)) in self.history :
                                self.history['candidates'+str(self.popcount(k))] += 1
                            else:
                                self.history['candidates'+str(self.popcount(k))] = 1

        for col in range(self.size):
            candsdict = dict()
            for (r,c) in self.columncells(0,col):
                if self.bitat(r,c) not in candsdict:
                    candsdict[self.bitat(r,c)] = list()
                candsdict[self.bitat(r,c)].append( (r,c) )
            for k in [t for t in candsdict if len(candsdict[t]) == self.popcount(t)]:
                for (r,c) in self.columncells(0,col):
                    if not self.isfixed(r,c) and (r, c) not in candsdict[k]:
                        bits = self.bitat(r,c)
                        self.bitput(r,c,self.bitat(r,c) & (all1^k))
                        if bits != 0 and self.bitat(r,c) == 0:
                            raise RuntimeError('Exhausted candidates.')
                        if bits != self.bitat(r,c) :
                            #print('col',r,c,bits,'->',self.bitat(r,c))
                            updated = True
                            if 'candidates'+str(self.popcount(k)) in self.history :
                                self.history['candidates'+str(self.popcount(k))] += 1
                            else:
                                self.history['candidates'+str(self.popcount(k))] = 1

        for row in range(0,self.size,self.factor):
            for col in range(0,self.size,self.factor):
                candsdict = dict()
                for (r,c) in self.blockcells(row,col):
                    if self.bitat(r,c) not in candsdict:
                        candsdict[self.bitat(r,c)] = list()
                    candsdict[self.bitat(r,c)].append( (r,c) )
                for k in [t for t in candsdict if len(candsdict[t]) == self.popcount(t)]:
                    for (r,c) in self.blockcells(row,col):
                        if not self.isfixed(r,c) and (r, c) not in candsdict[k]:
                            bits = self.bitat(r,c)
                            self.bitput(r,c,self.bitat(r,c) & (all1^k))
                            if bits != 0 and self.bitat(r,c) == 0:
                                raise RuntimeError('Exhausted candidates.')
                            if bits != self.bitat(r,c) :
                                #print('blk',r,c,bits,'->',self.bitat(r,c))
                                updated = True
                                if 'candidates'+str(self.popcount(k)) in self.history :
                                    self.history['candidates'+str(self.popcount(k))] += 1
                                else:
                                    self.history['candidates'+str(self.popcount(k))] = 1
        
        return updated

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

    def narrow(self):
        self.narrow_by_settled()
        while True:
            if self.narrow_by_candidates():
                continue
            break

    def guessed(self):
        guessed = list()
        for r in range(self.size):
            for c in range(self.size):
                if self.isfixed(r,c) :
                    continue
                for i in self.as_bitset(self.bitat(r,c)):
                    newsudoku = Sudoku(self)
                    newsudoku.put(r,c,i)
                    guessed.append(newsudoku)
                    if 'guessed' in newsudoku.history :
                        newsudoku.history['guessed'] += 1
                    else:
                        newsudoku.history['guessed'] = 1
        return guessed
    
if __name__ == '__main__':
    problems = ['000310008006080000090600100509000000740090052000000409007004020000020600400069000',
                '003020600900305001001806400008102900700000008006708200002609500800203009005010300',
                '615830049304291076000005081007000100530024000000370004803000905170900400000002003',
                '900000000700008040010000079000974000301080000002010000000400800056000300000005001',
                '400080100000209000000730000020001009005000070090000050010500400600300000004007603',
                '020000010004000800060010040700209005003000400050000020006801200800050004500030006',
                '001503900040000080002000500010060050400000003000201000900080006500406009006000300',
                '080100000000070016610800000004000702000906000905000400000001028450090000000003040',
                '001040600000906000300000002040060050900302004030070060700000008000701000004020500',
                '000007002001500790090000004000000009010004360005080000300400000000000200060003170',
                '001000000807000000000054003000610000000700000080000004000000010000200706050003000',
                '090000040000180000000050600000000094001000000020000070006000800000907000005000100',
                '300012400600000000000009200500300006000004000000000008800500000000600000002000900',
                '100000023004005001006001007020080000010060040000070090500900100600300800790000006',
                ]
    
    total = 0
    for p in problems:
        s = Sudoku(p)
        print(s)
        #solved = list()
        dt = datetime.datetime.now()
        frontier = deque([s])
        while bool(frontier):
            s = frontier.popleft()
            try:
                s.narrow()
            except RuntimeError:
                continue
            if s.issolved():
                break
            for e in s.guessed():
                frontier.append(e)
        delta = datetime.datetime.now() - dt
        millis = delta.seconds*1000+ delta.microseconds/1000
        print(millis)
        total += millis
        print(s)
        print(s.history)
    print('finished in {:.4f} millisec.'.format(total))
