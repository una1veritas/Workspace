
import glob
import csv
import sys
import math
from collections import deque
from statistics import stdev

if len(sys.argv) < 2:
    exit
stockcode = sys.argv[1]
print (stockcode)
files_list = glob.glob('../yfseries/'+stockcode+'-*-*.csv')
print(files_list)

tseries = []
header = []
for fname in files_list:
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # ヘッダーを読み飛ばしたい時
        if len(header) == 0 : 
            header = next(csv_reader)  # ヘッダーを読み飛ばしたい時
        else:
            next(csv_reader)
        for row in csv_reader:
            for index in range(1,len(row)):
                row[index] = int(row[index])
            tseries.append(row)

tseries.sort()

#moving average
avrspans = [ 5, 21, 55 ]
spanqs = [ deque([ ]), deque([ ]), deque([ ]) ]
header = header + ['m.avr.5', 'm.avr.21', 'm.avr.55']
for i in range(0,len(tseries)) :
    mavrs = [ 0, 0, 0 ]
    for sindex in range(0, len(avrspans)) :
        spanqs[sindex].append(tseries[i][4])
        if len(spanqs[sindex]) >= avrspans[sindex] :
            spanqs[sindex].popleft()
        mavrs[sindex] = round(sum(spanqs[sindex])/len(spanqs[sindex]),1)
    tseries[i] = tseries[i] + mavrs
    if len(spanqs[2]) > 1 :
        tseries[i].append(round(stdev(spanqs[2]),1))
    else:
        tseries[i].append(0)

#moving volume weighted average
avrspans = [ 5, 21, 55 ]
spanqs = [ deque([ ]), deque([ ]), deque([ ]) ]
volqs  = [ deque([ ]), deque([ ]), deque([ ]) ]
header = header + ['m.vwavr.5', 'm.vwavr.21', 'm.vwavr.55']
for i in range(0,len(tseries)) :
    vwmavrs = [ 0, 0, 0 ]
    for sindex in range(0, len(avrspans)) :
        spanqs[sindex].append(tseries[i][4])
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
#    if len(spanqs[2]) > 1 :
#                
#    else:
#        tseries[i].append(0)

#Bollinger band, volume weighted
span = 21
header = header + ['vw.stddev. 21']
for i in range(0,len(tseries)) :
    newrow = [ '' ]
    if i < span - 1:
        continue
    else:
        psum = 0
        vsum = 0
        for j in range(i-span+1, i+1) :
            psum = psum + tseries[j][4] * (tseries[j][5]/100)
            vsum = vsum + (tseries[j][5]/100)
        avr = round(psum / vsum, 1)
        dev2sum = 0
        for j in range(i-span+1, i+1) :
            dev2sum = dev2sum + (tseries[j][5]/100) * ((tseries[j][4] - avr)** 2)
        s = math.sqrt(dev2sum/(vsum-1))
        newrow[0] = round(s,1)
    tseries[i] = tseries[i] + newrow

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

