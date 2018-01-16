# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
import os
from pyquery import PyQuery

#if __name__ == '__main__':
#    pass

#default values
kd=1
mk=2
tm='d'
vl='a'
fpath='./'
# from argv
if len(sys.argv) == 5:
    kd = int(sys.argv[1])
    tm = str(sys.argv[2])
    mk = int(sys.argv[3])
    vl=sys.argv[4]
#print([kd, tm, mk, vl])

def yahooFinanceRanking(kind=1,timespan='d',market=2,volume='a',pages=5):
    ranking = []
    if kind == 33 or kind == 36:
        timespan = 'd'
    for page in range(1,pages+1):
        url = "https://info.finance.yahoo.co.jp/ranking/?kd={0}&tm={1}&mk={2}&vl={3}&p={4}".format(kind,timespan,market,volume,page)
        pquery = PyQuery(url)
        if page == 1:
            #<div class="ttl"><h1 class="inner">
            stamp_str = pquery(pquery('div.ttl')[0]).text().split(u'最終更新日時：')[1]
            [tyear, stamp_str] = stamp_str.split(u'年')
            [tmon, stamp_str] = stamp_str.split(u'月')
            [tdate, stamp_str] = stamp_str.split(u'日')
            [thour, stamp_str] = stamp_str.split(u'時')
            [tmin, stamp_str] = stamp_str.split(u'分')
            rowlist = [ '0', str(kind)+str(timespan)+str(market)+str(volume),
                       str(tyear).zfill(4)+str(tmon).zfill(2)+str(tdate).zfill(2), 
                       str(thour).zfill(2)+str(tmin).zfill(2) ]
            ranking.append(rowlist)
        #print(pq)
        for tr in pquery('tbody')('tr'):
            tr_row = pquery(tr)
            rowlist = []
            for td in tr_row('td'):
                tr_str = tr_row(td).text()
                if tr_str == u'掲示板' : continue
                if kind == 33 or kind == 36:
                    tr_str = tr_str.rstrip(u'倍')
                rowlist.append(tr_str.lstrip(u'+').replace(',',''))
            ranking.append(rowlist)
    return ranking

ranking = yahooFinanceRanking(kind=kd,timespan=tm,market=mk,volume=vl,pages=10)
print( '{0} ranking {1} on {2} updated at {3}'.format(ranking[0][0],ranking[0][1],ranking[0][2],ranking[0][3]))
f_name = 'yfr'+ranking[0][1]+'-'+ranking[0][2]+ranking[0][3]+'.csv'
#yahooFinanceRanking(ranking, timespan='w',page=2)

f_encode = 'sjis'
if not os.path.exists(fpath+f_name):
    with open(f_name,'w') as csv_file:
        for rowlist in ranking:
            colnum = len(rowlist)
            cnt = 0
            for item in rowlist:
                csv_file.write(item)
                cnt = cnt + 1
                if cnt < colnum:
                    csv_file.write(',')
            csv_file.write('\n')
    csv_file.close()
else:
    print ('file ' + f_name + ' already exists.')
print ('done.')

