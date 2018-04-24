'''
'''

import sys
import glob
#import math
#
import pandas as pd
#
import pattern as dspat

params = { 'pattern': '', 'files' : []}

args = sys.argv
del args[0]
while len(args) > 0:
    arg = args.pop(0)
    if arg[0] == '-':
        if '=' in arg[1:]:
            argval = arg[1:].split('=')
            argkey = argval[0]
            argval.pop(0)
        else:
            argkey = arg[1:]
            argval = args.pop(0)
        if 'pattern'.startswith(argkey) :
            argkey = 'pattern'
        params[argkey] = argval
    else:
        params['files'].append(arg)
#print(params)

if 'pf' in params and len(params['pattern']) == 0:
    filename = params['pf']
    pattstr = ''
    with open(filename) as pf:
        for a_line in pf:
            if a_line[0] == '#' : continue
            a_line = a_line.strip()
            if len(a_line) == 0 : continue
            pattstr = pattstr + a_line
            if a_line[-1] == ',' : continue
            break
    if len(pattstr) > 0 :
        params['pattern'] = pattstr

files = params['files']
if len(files) != 0:
    files.sort()

for fname in files:
    print(fname)
    tseries = pd.read_csv(fname, index_col='date') #, parse_dates=['date'] )
    tseries = tseries.append(pd.read_csv(fname, index_col='date')) #, parse_dates=['date'] )
    tseries.sort_index()
    tsseq = tseries[['open','high','low','close','volume']].reset_index().values
    
    patt = dspat.SequencePattern(params['pattern'])
    print('sequence pattern = '+str(patt))
    
    for pos in range(max(0,len(tsseq)-200), len(tsseq)):
        (result, vardict) = patt.match(tsseq, pos)
        if result :
            print(tsseq[pos : pos+patt.patternCount()+1])
            print(pos, vardict)
            print()

