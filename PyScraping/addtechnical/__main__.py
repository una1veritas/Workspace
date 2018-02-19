# add technical indexes and oscillators 

import glob
import sys
import math
#from collections import deque
from statistics import stdev
import pandas as pd
#import numpy as np
from operator import itemgetter

#
import matplotlib.pyplot as plt
import matplotlib.finance as mf
#from matplotlib import ticker
import matplotlib.dates as mdates
#from matplotlib.pyplot import tight_layout

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

# default values for options
params = { 'sma' : [5, 25, 50], 'path' : './data' }

for arg in sys.argv[1:] :
    if arg == '-vw' :
        params['volw'] = True
    elif arg[:6] == '-path=' :
        params['path'] = arg[6:].rstrip('/')
    elif arg[:2] == '-m' :
        t = arg.split('.')[1:]
        params['sma'] = [ int(t[0]), int(t[1]), int(t[2]) ]
    elif arg[:6] == '-bband' :
        t = arg.split('.')[1:]
        print(t)
        params['bband'] = int(t.pop(0))
    elif arg[:4] == '-rci' :
        t = arg.split('.')[1:]
        params['rci'] = int(t.pop(0))
#    elif arg == '-ohlc' :
#        params['ohlc'] = True
    elif arg == '-plot' :
        params['plot'] = True
    else:
        if len(arg.split('.')) == 2 :
            params['code'] = arg
        else:
            params['code'] = arg + '.T'

if not ('code' in params) :
    exit
    
print (params)
print(params['path'] + '/' + params['code']+'-*-*.csv')
files_list = glob.glob(params['path'] + '/' + params['code']+'-*-*.csv')
files_list.sort()
print(files_list)

tseries = pd.read_csv(files_list[0], index_col='date', parse_dates=['date'])
for fname in files_list[1:]:
    tseries = tseries.append(pd.read_csv(fname, index_col='date', parse_dates=['date']))

tseries.sort_index()

# normalize the prices
if 'adj.close' in tseries.columns:
    for i in range(0,len(tseries.index)) :
        adjfact = tseries['adj.close'].iat[i]/tseries['close'].iat[i]
        if adjfact != 1.0 :
            tseries.iat[i, 0] = tseries.iat[i,0] * adjfact
            tseries.iat[i, 1] = tseries.iat[i,1] * adjfact
            tseries.iat[i, 2] = tseries.iat[i,2] * adjfact
            tseries.iat[i, 3] = tseries.iat[i,3] * adjfact

# moving averages
def SimpleMovingAverages(dframe, avrspans):
    mavrs = { }
    for span in avrspans:
        mavrs['sma '+str(span)] = [ ]
    priceq = dframe['close']
    for span in avrspans:
        mavr = pd.rolling_mean(priceq,window=span, min_periods=1)
        mavrs['sma '+str(span)].append(round(mavr,1))
    for smaname in mavrs :
        dframe[smaname] = mavrs[smaname][0]
    return

#add moving averages
if 'sma' in params:
    SimpleMovingAverages(tseries, params['sma'])

if 'bband' in params:
    #add Bollinger band lines
    if 'adj.close' in tseries.columns :
        adjclose = list(tseries['adj.close'])
    else:
        adjclose = list(tseries['close'])
    span = int(params['bband'])
    mpv = list(tseries['sma '+str(span)])
    if 'volume' in tseries.columns :
        vol = list(tseries['volume'])
    else:
        vol = [ ]
    stddev = [ 0 ]
    bollband = [ [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]] ]
    for i in range(1,len(mpv)) :
        if 'volw' in params and len(vol) > 0:
            dev2sum = 0
            vsum = 0
            for i in range(max(0,i+1-params['sma'][1]), i+1) :
                dev2sum = dev2sum + vol[i] * (adjclose[i] - mpv[i])**2
                vsum = vsum + vol[i]
            sigma = math.sqrt(dev2sum/(vsum-1))
        else:
            sigma = stdev(adjclose[max(0,i+1-params['sma'][1]):i+1])
        stddev.append(round(sigma,1))
        bollband[0].append(round(mpv[i]-3*sigma,1))
        bollband[1].append(round(mpv[i]-2*sigma,1))
        bollband[2].append(round(mpv[i]+2*sigma,1))
        bollband[3].append(round(mpv[i]+3*sigma,1))
        
    tseries['stddev'] = stddev
    tseries['-3s'] = bollband[0]
    tseries['-2s'] = bollband[1]
    tseries['+2s'] = bollband[2]
    tseries['+3s'] = bollband[3]

if 'rci' in params:
    # add RCI oscillator
    rciq = []
    if 'adj.close' in tseries.columns :
        dateprice = list(zip(tseries.index.tolist(),tseries['adj.close'].tolist()))
    else:
        dateprice = list(zip(tseries.index.tolist(),tseries['close'].tolist()))    
    span = int(params['rci'])
    for ix in range(0, len(dateprice)) :
        dprange = dateprice[max(0,ix+1-span):ix+1]
        dprange.sort(key=itemgetter(1),reverse=True)
        dev2sum = 0
        for dx in range(0,len(dprange)):
            drank = dx + 1
            dp = dateprice[max(0,ix+1-span):ix+1][-drank]
            prank = dprange.index(dp) + 1
            dev2sum = dev2sum + (drank-prank)**2
        rci = (1 - 6 * dev2sum / (span*(span - 1)*(span+1))) * 100
    #    print(round(rci,1))
        rciq.append(round(rci,1))
    tseries['RCI'] = rciq

#output
tseries.to_csv(params['code']+'-'+'anal'+'.csv')

#plot
if 'plot' in params :
    df = tseries[-75:][[ 'open', 'high', 'low', 'close', 'volume']].reset_index()
    #df.columns = ["Date","Open","High",'Low',"Close"]
    df['date'] = df['date'].map(mdates.date2num)
    print(df)
    
    #Making plot
    fig = plt.figure(facecolor='#07000d')
    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
    #fig, ax = plt.figure(), plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)
    #fig, axes = plt.subplots(ncols=1,nrows=2, sharex=True, figsize=(4,6))
    #Converts raw mdate numbers to dates
    ax1.xaxis_date()
    plt.xlabel("Date")
    
    #Making candlestick plot
    #mfinance.candlestick_ohlc(axes[0],df.values,width=0.6, colorup='red', colordown='blue',alpha=1)
    mf.candlestick_ohlc(ax1, df, width=.6, colorup='#53c156', colordown='#ff1717')
    
    #sma = df['close'].rolling(5).mean()
    #vstack = np.vstack((range(len(sma)), sma.values.T)).T  # x軸データを整数に
    #ax.plot(vstack[:, 0], vstack[:, 1])
    
    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    ax0.grid(True) #グリッド表示
    fig.autofmt_xdate()
    #for label in ax1.xaxis.get_ticklabels():
    #    label.set_rotation(90)
    #fig,tight_layout()
    plt.ylabel("Price")
    #plt.legend()
    
    plt.show()


