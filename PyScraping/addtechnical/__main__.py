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
#import matplotlib.finance as mf
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
params = { 'sma' : [5, 25, 50], 'files' : [] }

args = sys.argv
del args[0]
if len(args) == 0 :
    print('-sma -path -mon -bband -rci files')
    exit()
    
for arg in args :
    if arg[0] == '-' :
        pname, pvalue = arg[1:].split('=')
        params[pname] = eval(pvalue)
    else:
        params['files'].append(arg)

print (params)
if 'path' in params :
    for i in range(len(params['files'])) :
        params['files'][i] = params['path'] + '/' + params['files'][i] 
#print(params['path'] + '/' + params['code']+'-*-*.csv')
#files_list = glob.glob(params['path'] + '/' + params['code']+'-*.csv')
files_list = params['files']
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
        mavr = priceq.rolling(window=int(span), min_periods=1).mean()
        mavrs['sma '+str(span)].append(round(mavr,1))
    for smaname in mavrs :
        dframe[smaname] = mavrs[smaname][0]
    return

#add moving averages
if 'sma' in params:
    SimpleMovingAverages(tseries, params['sma'])

if 'mom' in params:
    back = int(params['mom'])    
    momentum = []
    priceseq = list(tseries['close'])
    for i in range(0, len(priceseq)) :
        iback = max(0,i-back)
        momentum.append(priceseq[i] - priceseq[iback])
    tseries['momentum ('+str(back)+')'] = momentum
    
if 'bband' in params:
    #add Bollinger band lines
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
            for i in range(max(0,i+1-span), i+1) :
                dev2sum = dev2sum + vol[i] * (adjclose[i] - mpv[i])**2
                vsum = vsum + vol[i]
            sigma = math.sqrt(dev2sum/(vsum-1))
        else:
            sigma = stdev(adjclose[max(0,i+1-span):i+1])
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
if 'adj.close' in tseries.columns :
    tseries = tseries.drop(labels='adj.close', axis=1)
outfilename = ''
outfilename = files_list[-1].split('-')[0] + '-anal.csv'
tseries.to_csv(outfilename)

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
    #vstack = np.vstack((range(len(sma)), sma.values.T)).T  # x霆ｸ繝�繝ｼ繧ｿ繧呈紛謨ｰ縺ｫ
    #ax.plot(vstack[:, 0], vstack[:, 1])
    
    ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    ax0.grid(True) #繧ｰ繝ｪ繝�繝芽｡ｨ遉ｺ
    fig.autofmt_xdate()
    #for label in ax1.xaxis.get_ticklabels():
    #    label.set_rotation(90)
    #fig,tight_layout()
    plt.ylabel("Price")
    #plt.legend()
    
    plt.show()


