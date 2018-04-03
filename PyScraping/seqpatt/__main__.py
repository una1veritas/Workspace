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
            args.pop(0)
            argval = args.pop(0)
            print(argkey, argval)            
        params[argkey] = argval
    else:
        params['files'].append(arg)
        
print(params)

files = params['files']
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

patt = dspat.SequencePattern(params[1])
print('sequence seqpatt = '+str(patt))

for pos in range(max(0,len(tsseq)-200), len(tsseq)):
    (result, vardict) = patt.match(tsseq, pos)
    if result :
        print(tsseq[pos : pos+patt.patternCount()+1])
        print(pos, vardict)
        print()
    