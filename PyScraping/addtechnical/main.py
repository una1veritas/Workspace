# add technical indexes and oscillators 

import glob
import sys
import math
#from collections import deque
from statistics import stdev
import pandas as pd
from operator import itemgetter

#
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib import ticker
import matplotlib.dates as mdates

def candlechart(ohlc, width=0.8):
    """入力されたデータフレームに対してローソク足チャートを返す
        引数:
            * ohlc:
                *データフレーム
                * 列名に'open'", 'close', 'low', 'high'を入れること
                * 順不同"
            * widrh: ローソクの線幅
        戻り値: ax: subplot"""
    fig, ax = plt.subplots()
    # ローソク足
    mpf.candlestick2_ohlc(ax, opens=ohlc.open.values, closes=ohlc.close.values,
                          lows=ohlc.low.values, highs=ohlc.high.values,
                          width=width, colorup='w', colordown='k')

    # x軸を時間にする
    xdate = ohlc.index
    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))

    def mydate(x, pos):
        try:
            return xdate[int(x)]
        except IndexError:
            return ''

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

    fig.autofmt_xdate()
    fig.tight_layout()

    return fig, ax

# default values for options
params = { 'mavr' : [5, 25, 50] }

for arg in sys.argv[1:] :
    if arg == '-vw' :
        params['volw'] = True
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
files_list.sort()
print(files_list)

tseries = pd.read_csv(files_list[0], index_col= 0)
for fname in files_list[1:]:
    tseries = tseries.append(pd.read_csv(fname, index_col = 0))

tseries.sort_index()

#add moving averages
avrspans = params['mavr']
priceq = [ ]
volumeq  = [ ]
mavrs = { }
# moving averages
for span in avrspans:
    mavrs['avr.'+str(span)] = [ ]
for i in range(0,len(tseries.index)) :
    adjfact = tseries['adj.close'].iat[i]/tseries['close'].iat[i]
    if adjfact != 1.0 :
        tseries.iat[i, 0] = tseries.iat[i,0] * adjfact
        tseries.iat[i, 1] = tseries.iat[i,1] * adjfact
        tseries.iat[i, 2] = tseries.iat[i,2] * adjfact
        tseries.iat[i, 3] = tseries.iat[i,3] * adjfact
    if 'ohlc' in params :
        price =  round(sum(list(tseries.iloc[i][0:4]))/4,1)
    else:
        price = tseries['close'].iat[i]
    priceq.append(price)
    for span in avrspans:
        if 'volw' in params :
            psum = 0
            vsum = 0
            for idx in range(max(0,i+1-span), i+1) :
                vol = tseries['volume'].iat[idx]
                vsum = vsum + vol
                psum = psum + priceq[idx] * vol
            mavr = psum/vsum
        else:
            mavr = sum(priceq[max(0,i+1-span):i+1])/(i + 1 - max(0, i+1-span))
        mavrs['avr.'+str(span)].append(round(mavr,1))
for span in avrspans :
    tseries['avr.'+str(span)] = mavrs['avr.'+str(span)]    

#add Bollinger band lines
adjclose = list(tseries['adj.close'])
mpv = list(tseries['avr.'+str(avrspans[1])])
vol = list(tseries['volume'])
stddev = [ 0 ]
bollband = [ [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]], [mpv[0]] ]
for i in range(1,len(mpv)) :
    if 'volw' in params :
        dev2sum = 0
        vsum = 0
        for i in range(max(0,i+1-avrspans[1]), i+1) :
            dev2sum = dev2sum + vol[i] * (adjclose[i] - mpv[i])**2
            vsum = vsum + vol[i]
        sigma = math.sqrt(dev2sum/(vsum-1))
    else:
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

# add RCI oscillator
rciq = []
dateprice = list(zip(tseries.index.tolist(),tseries['adj.close'].tolist()))
span = avrspans[1]
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
candlechart(tseries[-120:],width=1.0)
plt.show()