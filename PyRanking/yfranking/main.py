# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
from pyquery import PyQuery

#if __name__ == '__main__':
#    pass

#default values
kind=33
market=2
timespan='d'
# from argv
if len(sys.argv) == 4:
    kind = int(sys.argv[1])
    timespan=sys.argv[2]
    market = int(sys.argv[3])

def yahooFinanceRanking(yfrkind=33,yfrmarket=2,yfrtimespan='d',yfrvolume='a'):
    ranking = []
    for page in range(1,5):
        url = "https://info.finance.yahoo.co.jp/ranking/?kd={0}&tm={1}&vl={3}&mk={2}&p={4}".format(yfrkind,yfrtimespan,yfrmarket,yfrvolume,page)
        pquery = PyQuery(url)
        if page == 1:
            rowlist = [ 0 ]
            for spans in pquery('div.ttl'): #<div class="ttl"><h1 class="inner">
                print( pquery(spans).text().split(u'最終更新日時：')[1] )
        #print(pq)
        for tr in pquery('tbody')('tr'):
            tr_row = pquery(tr)
            rowlist = []
            if yfrkind == 33:
                for td in tr_row('td'):
                    rowlist.append(tr_row(td).text().rstrip(u'倍').replace(',',''))
            elif yfrkind == 1:
                for td in tr_row('td'):
                    rowlist.append(tr_row(td).text().lstrip(u'+').replace(',',''))
            ranking.append(rowlist)
    return ranking

ranking = yahooFinanceRanking(yfrkind=kind,yfrmarket=market,yfrtimespan=timespan)
f_name = 'pyquery.csv'
#yahooFinanceRanking(ranking, timespan='w',page=2)

f_encode = 'sjis'
#if not os.path.exists(filepath+f_name):
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
#else:
#    print ('file ' + f_name + ' already exists.')
print ('done.')

