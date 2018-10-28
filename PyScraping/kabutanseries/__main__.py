# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
import math
import pandas
from bs4 import BeautifulSoup
import urllib.request
#import csv

if __name__ == '__main__':
    pass

#https://kabutan.jp/stock/kabuka?code=0000
#TOPIX: 0010
#NIKKEI225: 0000
#NYDOWJ: 0800
#https://kabutan.jp/stock/kabuka?code=0000&ashi=day&page=2
params = { 'path': './', 'code' : '0010', 
          'fromdate' : 20160101, 'todate' : 20181231, 'interval' : 'daily'}

def get_params(args, default_params):
    params = dict(default_params)
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
            if 'fromdate'.startswith(argkey) :
                argkey = 'fromdate'
            elif 'todate'.startswith(argkey) :
                argkey = 'todate'
            elif 'interval'.startswith(argkey) :
                argkey = 'interval'
            elif argkey.endswith('span') :
                argkey = 'timespan'
            params[argkey] = argval
        else:
            params['codes'].append(arg)
    return params

params = get_params(sys.argv, params)
print(params)


#timestamp = datetime.now().strftime("%Y%m%d-%H%M")
#print ("current time stamp: ", timestamp)
# 1カラム目に時間を挿入します
#rowlist.append(timestamp)

def CalDate(jd) :
    jd = jd + 0.5
    z = int(jd)
    a = z
    f = jd - int(jd)
    if ( z >= 2299161 ) :
        alpha = int( (z-1867216.25)/36524.25 )
        a = a + 1 + alpha - int(alpha/4)
    b = a + 1524
    c = int( (b-122.1)/365.25 )
    d = int(365.25 * c)
    e = int( (b-d)/30.6001 )
    date = b - d - int(30.6001 * e) + f
    if ( e < 13.5 ) :
        month = e - 1
    else:
        month = e-13
    if ( month > 2.5) :
        year = c - 4716
    else:
        year = c - 4715
    return math.copysign(1,year)*(math.fabs(year)*10000 + month*100 + date)

def JulianDay(year, month, date):
    if ( month <= 2 ) :
        month = month + 12
        year = year - 1
    
    a = 0
    b = 0
    if ( year*10000+month*100+date >= 15821015 ) :
        a = int(year/100)
        b = 2-a+int(a/4)
    return int(365.25 * year) + int(30.6001 * (month+1)) + date + b + 1720994.5;

def get_kabutanTimeSeries(code, interval, fromdate, todate):
    url_template = 'https://kabutan.jp/stock/kabuka?code={0}&ashi={1}&page={2}'
    if interval.startswith('w') :
        interval = 'wek'
    elif interval.startswith('m') :
        interval = 'mon'
    else:
        interval = 'day'
    result = list()
    for page in range(1,2+1):
        #print('page = ' + str(page) )
        url = url_template.format(code, interval, page)
        #htmlsrc = urllib.request.urlopen(url).read().decode('utf-8') 
        ts_in_page = pandas.read_html(url, flavor='bs4')
        if page == 1 :
            #result.append(ts_in_page[13].iloc[0] )
            ts_in_page[12].columns=ts_in_page[13].iloc[0]
            ts_in_page[12].set_index('日付', inplace=True)
            result.append(ts_in_page[12][1:2])
        ts_in_page[13].columns = ts_in_page[13].iloc[0]
        ts_in_page[13].set_index('日付', inplace=True)
        result.append(ts_in_page[13][1:])
    return result
# tables = pd.read_html('http://stocks.finance.yahoo.co.jp/stocks/history/?code=998407.O', flavor='bs4')
# print(tables[1])        

result = get_kabutanTimeSeries(params['code'], params['interval'], params['fromdate'], params['todate'])
print(result[0].info())
print(result[1].info())
print(result[2].info())
exit()

header = ['date','open','high','low','close','volume'] 
table = kabutanTimeSeries(params['code'])

table = sorted(table)
lastdate = table[-1][0].replace('/','')
colnum = len(table[0])
df = pd.DataFrame(table,columns=header[:colnum])
df['code'] = params['code']
df = df[['code','date','open','high','low','close','volume']]
df = df.set_index(['code','date'])
fname = params['code']+'-'+lastdate+'.csv'
df.to_csv(fname)
print('csv file ' + fname + ' has been written.')
#dframe = pd.DataFrame[table, ]

#for row in table:
#    for index in range(0,len(row)):
#        print(row[index].replace(',',''), end='')
#        if index+1 < len(row): 
#            print(',', end='')
#        else:
#            print()
#yahooFinanceRanking(ranking, timespan='w',page=2)

