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

params = { 'path': './', 'tmspan': 'd'}
#code = '5698.T'
#pdstart = int('20171210')
#pdend = int('20180112')
#tmspan = 'd'
for argv in sys.argv[1:] :
    if argv[0] != '-' :
        if not 'code' in params :
            params['code'] = argv
        elif not 'fromdate' in params:
            params['fromdate'] = int(argv)
        elif not 'todate' in params:
            params['todate'] = int(argv)
        elif not 'path' in params :
            params['path'] = argv
    else:
        paramname, paramvalue = argv[1:].split('=')
        params[paramname] = paramvalue

print(params)
if not 'code' in params or not 'fromdate' in params or not 'todate' in params:
    print('code.m frmdate todate path')
    exit()
        
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

def yahooFinanceTimeSeries(code,pdstart,pdend,timespan='d'):
    pdstart = int(pdstart)
    pdend = int(pdend)
    url = 'https://info.finance.yahoo.co.jp/history/?code={0}&sy={1}&sm={2}&sd={3}&ey={4}&em={5}&ed={6}&tm={7}'
    url = url.format(code,pdstart//10000,pdstart // 100 % 100, pdstart % 100,
                     pdend//10000,pdend//100 % 100, pdend % 100,timespan)
    rows = [ ]
    events = [ ]
    pyquery = PyQuery(url)
    for table in pyquery('div.padT12')('table'):
        pytable = pyquery(table)
        for tr in pytable('tr'):
            row = pytable(tr)
            #skip header line
            if len(row('td')) == 0: 
                continue
            columns = []
            colnum = 0
            for td in row('td'):
                td_str = row(td).text()
                if colnum == 0 : 
                    td_date = td_str.replace(u'年','/').replace(u'月','/').replace(u'日','').split('/')
                    td_str = str(td_date[0]).zfill(4)+'/'+str(td_date[1]).zfill(2)+'/'+str(td_date[2]).zfill(2)
                    columns.append(td_str)
                elif u'分割' in td_str : 
                    events.append([ columns[0], td_str])
                    columns.clear()
                    break
                else:
                    td_str = td_str.replace(',','')
                    if '.' in td_str :
                        columns.append(float(td_str))
                    else:
                        columns.append(int(td_str))
                colnum = colnum + 1
            if len(columns) != 0 :
                rows.append(columns)
    return rows
#    for key in ranking:
#        if ranking[key][1] != u'東証ETF':
#            print key,": ",ranking[key]

jpstart = int(0.5+JulianDay(params['fromdate']//10000, params['fromdate']//100%100, params['fromdate']%100))
jpend = int(0.5+JulianDay(params['todate']//10000, params['todate']//100%100, params['todate']%100))
table = [ ]
header = ['date','open','high','low','close','volume','adj.close'] 
for jd in range(jpstart, jpend+1, 32):
    if jd+31 > jpend:
        je = jpend
    else:
        je = jd+31
    pjstart = int(CalDate(jd))
    pjend = int(CalDate(je))
    table = table + yahooFinanceTimeSeries(params['code'], pjstart, pjend, params['tmspan'])

table = sorted(table, reverse=False)
colnum = len(table[0])
df = pd.DataFrame(table,columns=header[:colnum])
df = df.set_index('date')
basepath = params['path'].rstrip('/')
if basepath != '' :
    basepath = basepath + '/'
df.to_csv(basepath + params['code']+'-'+str(params['fromdate'])+'-'+str(params['todate'])+'.csv')
#dframe = pd.DataFrame[table, ]

#for row in table:
#    for index in range(0,len(row)):
#        print(row[index].replace(',',''), end='')
#        if index+1 < len(row): 
#            print(',', end='')
#        else:
#            print()
#yahooFinanceRanking(ranking, timespan='w',page=2)

