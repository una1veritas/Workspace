'''
'''

import sys
import glob
#import math
#
import pandas as pd
#
import pattern as dspat

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
filepatt = params['path'] + '/' + params[0]
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

patt = dspat.SequencePattern(params[1])
print('sequence seqpatt = '+str(patt))

for pos in range(max(0,len(tsseq)-200), len(tsseq)):
    (result, vardict) = patt.match(tsseq, pos)
    if result :
        print(tsseq[pos : pos+patt.patternCount()+1])
        print(pos, vardict)
        print()
    