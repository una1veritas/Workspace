
import glob
import csv
import sys
import math
from collections import deque
from statistics import stdev
import pandas as pd

# default values for options
params = { 'mavr' : [5, 25, 50] }

for arg in sys.argv[1:] :
    if arg == '-vw' :
        params['volweighted'] = True
    elif arg[:2] == '-m' :
        t = arg.split('.')[1:]
        params['mavr'] = [ int(t[0]), int(t[1]), int(t[2]) ]
    elif arg == '-ohlc' :
        params['ohlc'] = True
    else:
        if len(arg.split('.')) == 2 :
            params['code'] = arg
        else:
            params['code'] = arg + '.T'

if not ('code' in params) :
    exit
    
print (params)

files_list = glob.glob(params['code']+'-*-*.csv')
print(files_list)

# tseries = []
# header = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj.close' ]
# for fname in files_list:
#     with open(fname, 'r') as file:
#         csv_reader = csv.reader(file)
#         next(csv_reader)  # ヘッダーを読み飛ばしたい時
#         next(csv_reader)
#         for row in csv_reader:
#             for index in range(1,len(row)):
#                 row[index] = int(row[index])
#             tseries.append(row)
# 
# tseries.sort()
tseries = pd.read_csv(files_list[0], index_col= 0)
for fname in files_list[1:]:
    tseries = tseries.append(pd.read_csv(fname, index_col = 0))

print(tseries)

header = []
avrspans = params['mavr']
spanqs = [ deque([ ]), deque([ ]), deque([ ]) ]
header = header + ['m.avr.'+str(avrspans[0]), 'm.avr.'+str(avrspans[1]), 'm.avr.'+str(avrspans[2]), 'm.stddev']
if not ('volweighted' in params) :
    # moving average and std. deviations
    for i in range(0,len(tseries)) :
        mavrs = [ 0, 0, 0 ]
        for sindex in range(0, len(avrspans)) :
            if 'ohlc' in params :
                price = round(sum(tseries[i][1:4])/4,1)
            else:
                price = tseries[i][4]
            spanqs[sindex].append(price)
            if len(spanqs[sindex]) >= avrspans[sindex] :
                spanqs[sindex].popleft()
            mavrs[sindex] = round(sum(spanqs[sindex])/len(spanqs[sindex]),1)
        tseries[i] = tseries[i] + mavrs
        if len(spanqs[2]) > 1 :
            tseries[i].append(round(stdev(spanqs[2]),1))
        else:
            tseries[i].append(0)
else:
    # moving volume weighted average, and std. dev.
    volqs  = [ deque([ ]), deque([ ]), deque([ ]) ]
    for i in range(0,len(tseries)) :
        vwmavrs = [ 0, 0, 0 ]
        for sindex in range(0, len(avrspans)) :
            if 'ohlc' in params :
                price = round(sum(tseries[i][1:5])/4,1)
            else:
                price = tseries[i][4]
            spanqs[sindex].append(price)
            volqs[sindex].append(tseries[i][5]/100)
            if len(spanqs[sindex]) >= avrspans[sindex] : 
                spanqs[sindex].popleft()
                volqs[sindex].popleft()
            psum = 0
            vsum = 0
            for j in range(0,len(spanqs[sindex])) :
                psum = psum + spanqs[sindex][j]*volqs[sindex][j]
                vsum = vsum + volqs[sindex][j]
            vwmavrs[sindex] = round(psum/vsum, 1)
        tseries[i] = tseries[i] + vwmavrs
        if len(spanqs[1]) == 1 :
            sigma = 0
        else:
            vwmavr = vwmavrs[1]
            dev2sum = 0
            vsum = 0
            for j in range(0, len(spanqs[1])) :
                dev2sum = dev2sum + volqs[1][j] * ((spanqs[1][j] - vwmavr)**2) 
                vsum = vsum + volqs[1][j]
            sigma = math.sqrt(dev2sum / (vsum - 1))
        tseries[i].append(round(sigma,1))
        
for colname in header:
    print(str(colname)+',',end='')
print()

for row in tseries:
    print(row[0], end='')
    for col in row[1:] :
        print(',' + str(col), end='')
    print()

#for code in ranking:
#    print(code,': ',ranking[code],'  ',codedict[code])

#    tbl = pandas.read_csv(file,skiprows=1,header=None)
#    tbl = tbl.drop(9,axis=1)
#    tbl.columns = ['rank', 'code', 'market', 'name', 'date', 'price', 'ratio', 'diff', 'volume' ]
#    if count == 1 :
#        ranking = tbl.ix[:,['code', 'rank']]
#        ranking = ranking.sort_values(by='code')

