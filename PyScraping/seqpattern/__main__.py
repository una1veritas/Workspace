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

seqpattern = parsepattern(params[1])
print('sequence pattern = '+str(seqpattern))

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
                result = False
                subs.clear()
                break
        else:
            subs[patclause[lit]] = seqtuple[lit]
    return (result, subs)

def eval_opr(opr, vvlist, assigns):
    result = False
    values = [ ]
    for vname in vvlist:
        if vname in assigns :
            values.append(assigns[vname])
        else:
            values.append(float(vname))
    if opr == '<' or opr == '<=' or opr == '>' or opr == '>=' :
        result = True
        for val in values[1:] :
            if opr == '<' :
                result = result and (values[0] < val)
            elif opr == '>' :
                result = result and (values[0] > val)
            elif opr == '<=' :
                result = result and (values[0] <= val)
            elif opr == '>=' :
                result = result and (values[0] >= val)
            if not result : break
    elif opr == 'in' :
        result = True
        valrange = [min(values[0], values[1]),max(values[0], values[1])]
        for val in values[2:] :
            result = result and (valrange[0] <= val and val <= valrange[1]) 
            if not result : 
                result = False
                break
    else:
        print('opr. err.')
    return result, { }
    
for rootpos in range(max(0,len(tsseq)-200), len(tsseq)):
    assignments = { }
    flag = True
    tuples = 0
    for i in range(0,len(seqpattern)):
        if seqpattern[i][0][:1] == '@' :
            opr = seqpattern[i][0][1:]
            res, subs = eval_opr(opr, seqpattern[i][1:], assignments)
        else:
            if rootpos + tuples >= len(tsseq):
                flag = False
                break
            res, subs = match_clause(seqpattern[i], tsseq[rootpos+tuples], assignments)
            tuples = tuples + 1
        if not res:
            flag = False
            break
        for k in subs:
            assignments[k] = subs[k]
    if flag:
        print('pos at ' + str(rootpos) +', ' + str(assignments))
        print(tsseq[rootpos:min(rootpos+tuples+1,len(tsseq))])
