'''
'''

import sys
import glob
#import math
#
import pandas as pd
#

def parsepattern(patstr):
    patstr = patstr.lstrip().rstrip()
    pattern = [ ]
    ix = 0
    while ix < len(patstr) :
        leftix = patstr.find('[', ix)
        if ix == -1 : break
        rightix = patstr.find(']', ix + 1)
        clausestr = patstr[leftix:rightix+1]
        clauselist = clausestr[1:-1].split(',')
        pattern.append(clauselist)
        ix = rightix + 1
    return pattern

params = { 'path': ''}

args = sys.argv
del args[0]
argvcount = 0
while len(args) > 0:
    arg = args.pop(0)
    if arg[0] == '-':
        argval= arg[1:].split('=')
        argkey = argval[0]
        if len(argval) > 1 :
            argval = argval[1]
        else:
            argval = args.pop(0)
        params[argkey] = argval
    else:
        params[argvcount] = arg
        argvcount = argvcount + 1
        
print(params)

params['path'] = params['path'].rstrip('/')
if len(params['path']) == 0:
    params['path'] = '.'
filepatt = params['path'] + '/' + params[0] +'-*-*.csv'
print(filepatt)
files = glob.glob(filepatt)
if len(files) == 0:
    print('No files found.')
    exit()
files.sort()
print(files)

tseries = pd.read_csv(files.pop(0), index_col='date') #, parse_dates=['date'] )
for fname in files:
    tseries = tseries.append(pd.read_csv(fname, index_col='date')) #, parse_dates=['date'] )

tseries.sort_index()
tsseq = tseries[['open','high','low','close','volume']].reset_index().values

pattseq = parsepattern(params[1])
print(pattseq)

<<<<<<< Updated upstream
def match_clause(pclause, stuple, assigns):
    result = True
    subs = { }
    for lit in range(len(pclause)):
        if pclause[lit] == '*' : continue
        if pclause[lit] in assigns:
            if assigns[pclause[lit]] != stuple[lit] :
                result = False
                subs.clear()
                break
        elif pclause[lit] in subs:
            if subs[pclause[lit]] != stuple[lit] :
=======
def match_clause(patclause, seqtuple, assigns):
    result = True
    subs = { }
    for lit in range(len(patclause)):
        if patclause[lit] == '*' : continue
        if patclause[lit] in assigns:
            if assigns[patclause[lit]] != seqtuple[lit] :
                result = False
                subs.clear()
                break
        elif patclause[lit] in subs:
            if subs[patclause[lit]] != seqtuple[lit] :
>>>>>>> Stashed changes
                result = False
                subs.clear()
                break
        else:
<<<<<<< Updated upstream
            subs[pclause[lit]] = stuple[lit]
    return (result, subs)

assigns = { }
mlist = [ ]
for rootpos in range(0,min(5, len(tsseq))):
    mlist.clear()
    assigns.clear()
    mlist.append([rootpos, {}])
=======
            subs[patclause[lit]] = seqtuple[lit]
    return (result, subs)

assigns = { }
plist = [ ]
for pos in range(0,min(5, len(tsseq))):
    plist.clear()
    plist.append([pos, {}])
    cindex = 0
>>>>>>> Stashed changes
    while True:
        if len(mlist) > len(pattseq) :
            break
        lastpair = mlist[-1]
        cindex = len(mlist) - 1
        print(lastpair,cindex)
        res, subst = match_clause(pattseq[cindex], tsseq[lastpair[0]], lastpair[1])
        print(res, subst)
        if res :
            mlist[-1][1] = subst
            assigns = assigns + subst
            mlist.append([cindex+1, {}])
        else :
            mlist.pop()
            break
    if len(mlist) == len(pattseq) :
        print(mlist)
