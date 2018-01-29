
import glob
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

tseries.sort_index()

avrspans = params['mavr']
priceq = [ ]
volumeq  = [ ]
header = [ 'avr.'+str(avrspans[0]), 'avr.'+str(avrspans[1]), 'avr.'+str(avrspans[2])]
mavrs = { }
for spanname in header :
    mavrs[spanname] = [ ]
# moving average and std. deviations
for i in range(0,len(tseries.index)) :
    adjfact = tseries['adj.close'].iloc[i]/tseries['close'].iloc[i]
    if adjfact != 1.0 :
        tseries.iat[i, 0] = tseries.iat[i,0] * adjfact
        tseries.iat[i, 1] = tseries.iat[i,1] * adjfact
        tseries.iat[i, 2] = tseries.iat[i,2] * adjfact
        tseries.iat[i, 3] = tseries.iat[i,3] * adjfact
    if 'ohlc' in params :
        price =  round(sum(list(tseries.iloc[i][0:4]))/4,1)
    else:
        price = tseries['close'].iloc[i]
    priceq.append(price)
    for span in avrspans:
        mavr = sum(priceq[max(0,i+1-span):i+1])/(i + 1 - max(0, i+1-span))
        mavrs['avr.'+str(span)].append(round(mavr,1))
for colname in header :
    tseries[colname] = mavrs[colname]    

#print(tseries)
#             psum = 0
#             vsum = 0
#             for j in range(0,len(spanqs[sindex])) :
#                 psum = psum + spanqs[sindex][j]*volqs[sindex][j]
#                 vsum = vsum + volqs[sindex][j]
#             vwmavrs[sindex] = round(psum/vsum, 1)
#         tseries[i] = tseries[i] + vwmavrs
#         tseries[i].append(round(sigma,1))

adjclose = list(tseries['adj.close'])
mpv = list(tseries['avr.'+str(avrspans[1])])
vol = list(tseries['volume'])
stddev = [ 0 ]
bollband = [ [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]] ]
for i in range(1,len(mpv)) :
    sigma = stdev(adjclose[max(0,i+1-avrspans[1]):i+1])
    stddev.append(round(sigma,1))
    bollband[0].append(round(mpv[i]-3*sigma,1))
    bollband[1].append(round(mpv[i]-2*sigma,1))
    bollband[2].append(round(mpv[i]-1*sigma,1))
    bollband[3].append(round(mpv[i]+1*sigma,1))
    bollband[4].append(round(mpv[i]+2*sigma,1))
    bollband[5].append(round(mpv[i]+3*sigma,1))
    
tseries['stddev'] = stddev
tseries['-3s'] = bollband[0]
tseries['-2s'] = bollband[1]
tseries['-1s'] = bollband[2]
tseries['+1s'] = bollband[3]
tseries['+2s'] = bollband[4]
tseries['+3s'] = bollband[5]
# if len(spanqs[2]) > 1 :
#     mavrs[header[4]].append(round(stdev(spanqs[2]),1))
# else:
#     mavrs[header[4]].append(0.0)
#         if len(spanqs[1]) == 1 :
#             sigma = 0
#         else:
#             vwmavr = vwmavrs[1]
#             dev2sum = 0
#             vsum = 0
#             for j in range(0, len(spanqs[1])) :
#                 dev2sum = dev2sum + volqs[1][j] * ((spanqs[1][j] - vwmavr)**2) 
#                 vsum = vsum + volqs[1][j]
#             sigma = math.sqrt(dev2sum / (vsum - 1))

tseries.to_csv(params['code']+'-'+'anal'+'.csv')
