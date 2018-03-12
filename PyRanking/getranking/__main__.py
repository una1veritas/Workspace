# coding: utf-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
import os
from pyquery import PyQuery

if __name__ == '__main__':
    pass #特になし

# from argv
if len(sys.argv) <= 2:
    print('usage: basedirname rankingcode')
    exit()

if tuple(sys.version_info)[0] != 3:
    print('python 3.x required.')
    exit()    

def yahooFinanceRanking(kind=1,timespan='d',market=2,volume='a',pages=5):
    ranking = []
    if kind == 33 or kind == 36:
        timespan = 'd'
    print('reading ranking page ', end='', flush=True)
    for page in range(1,pages+1):
        print(str(page)+', ', end='', flush=True)
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
                if kind == 33 or kind == 36:
                    tr_str = tr_str.rstrip(u'倍')
                tr_str = tr_str.lstrip(u'+').rstrip('%').strip()
                tr_str = tr_str.replace(',','')
                rowlist.append(tr_str)
            if len(rowlist) == 10:
                rowlist = rowlist[0:4]+rowlist[5:9]
            else:
                rowlist = rowlist[0:8]
            ranking.append(rowlist)
    print()
    return ranking

def save_ranking(basedir, kind, timespan, market, volume, pages):
    ranking = yahooFinanceRanking(kind,timespan,market,volume,pages)
    print( 'ranking {0} on {1} updated at {2}'.format(ranking[0][1],ranking[0][2],ranking[0][3]))
    f_name = basedir + '/'+'yfr'+ranking[0][1]+'-'+ranking[0][2]+ranking[0][3]+'.csv'
    #yahooFinanceRanking(ranking, timespan='w',page=2)
    if os.path.exists(f_name):
        print ('overwriting file ' + f_name + '.')
    with open(f_name, mode='w', encoding='utf-8') as csv_file:
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
    return

#

params = { 'path' : '.', 'codes': [] }
for arg in sys.argv[1:] :
    if arg[0] == '-' :
        values = arg[1:].split('=')
        if len(values) == 1:
            params[values[0]] = True
        else:
            params[values[0]] = values[1]
    else:
        params['codes'].append(arg)

if not os.path.exists(params['path']) :
    print('path \"' + params['path'] + '\" does not exist.')
    exit()
    
for rkcode in params['codes']:
    print(rkcode)
    if rkcode == '1d2a' :
        kd = 1
        mk = 2
        tm = 'd'
        vl = 'a'
    elif rkcode == '1w2a' :
        kd = 1
        mk = 2
        tm = 'w'
        vl = 'a'
    elif rkcode == '2d2a' :
        kd = 2
        mk = 2
        tm = 'd'
        vl = 'a'
    elif rkcode == '2w2a' :
        kd = 2
        mk = 2
        tm = 'w'
        vl = 'a'
    elif rkcode == '33d2a' :
        kd = 33
        mk = 2
        tm = 'd'
        vl = 'a'
    elif rkcode == '36d2a' :
        kd = 36
        mk = 2
        tm = 'd'
        vl = 'a'
    else:
        print('unknown code '+rkcode)
    save_ranking(params['path'],kd,tm,mk,vl,36)

print ('done.')
exit()
