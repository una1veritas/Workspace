# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
import math
from pyquery import PyQuery
import pandas as pd

#import csv

if __name__ == '__main__':
    pass

#https://kabutan.jp/stock/kabuka?code=0000
#https://kabutan.jp/stock/kabuka?code=0000&ashi=day&page=2
params = { 'path': './', 'code' : '0000', 'url_base' : 'https://kabutan.jp/stock/kabuka?', 
          'fromdate' : 20170101, 'todate' : 20181229, 'tmspan' : 'day'}

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

def kabutanTimeSeries(code,timespan='day'):
    urltemplate = params['url_base'] + 'code={0}&ashi={1}&page={2}'
    hist = [ ]
    for i in range(10):
        url = urltemplate.format(code,timespan,i+1)
        print('url='+url)
        pyquery = PyQuery(url)
        if len(hist) == 0 :
            table_today = pyquery('table.stock_kabuka0')
            today = [ ]
            col_index = 0
            for td in pyquery(list(table_today('tr'))[1])('td'):
                td_str = pyquery(td).text()
                if len(today) == 0 :
                    td_str = '20'+td_str
                    today.append(td_str)
                elif not (col_index == 5 or col_index == 6):
                    td_str = td_str.lstrip('+').replace(',','')        
                    today.append(float(td_str))
                col_index = col_index + 1
            hist.append(today)
                    
        table_history = pyquery('table.stock_kabuka1')
        for tr in pyquery(table_history)('tr'):
            row = table_history(tr)
            col_list = []
            col_index = 0
            for td in row('td'):
                td_str = row(td).text()
                if len(col_list) == 0 :
                    td_str = '20'+ td_str
                    col_list.append(td_str)
                elif not (col_index == 5 or col_index == 6):
                    td_str = td_str.lstrip('+').replace(',','')
                    col_list.append(float(td_str))
                col_index = col_index + 1
            if len(col_list) == 0:
                continue
            hist.append(col_list)
    return hist
#    for key in ranking:
#        if ranking[key][1] != u'東証ETF':
#            print key,": ",ranking[key]

header = ['date','open','high','low','close','volume'] 
table = kabutanTimeSeries(params['code'])

table = sorted(table)
lastdate = table[-1][0].replace('/','')
colnum = len(table[0])
df = pd.DataFrame(table,columns=header[:colnum])
df = df.set_index('date')
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

