# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
#import csv
#import time

if __name__ == '__main__':
    pass

ranking_kind=1
ranking_timespan='w'
ranking_market=2
ranking_lastpage=1
if len(sys.argv) == 5:
    ranking_kind = int(sys.argv[1])
    ranking_timespan = sys.argv[2]
    ranking_market = int(sys.argv[3])
    ranking_lastpage = int(sys.argv[4])
        
filepath = ''
tstamp = datetime.now().strftime("%Y%m%d-%H%M")
print ("current time stamp: ", tstamp)
# 1カラム目に時間を挿入します
#rowlist.append(timestamp)

def yahooFinanceRanking(kind=1, timespan='d', market=1, lastpage=1):
#    ranking_kind = { 1:u'値上がり率', 33:u'出来高増加率' }
#    ranking_market = { 1:u'全市場', 2:u'東証全体', 3:u'東証1部', 4:u'東証2部', 5:u'東証マザーズ', 6:u'JASDAQ全体', 6:u'JASDAQスタンダート', }
#    ranking_timespan = { 'd':'日次', 'w':u'週次', 'm':u'月次' }

    ranking = {}
    target_page = 1
    while target_page <= lastpage:
        url = "https://info.finance.yahoo.co.jp/ranking/?kd={0}&tm={1}&vl=a&mk={2}&p={3}".format(kind,timespan,market,target_page)
        bsoup = BeautifulSoup(requests.get(url).content, "html.parser")
        if target_page == 1:
            div_ttl = bsoup.find("div", {"class" : "ttl"})
            datestr = str(div_ttl.find("div"))#,"utf-8"
            datestr = datestr.split(u'：')[1].split('</div>')[0]
            datestr = datestr.replace(u'年','/')
            datestr = datestr.replace(u'月','/')
            datestr = datestr.replace(u'日','-')
            datestr = datestr.replace(u'時',':')
            datestr = datestr.replace(u'分','')
            print ("time stamp of the ranking: ",datestr)
            tmp_date = datestr.split('-')[0].split('/')
            tmp_time = datestr.split('-')[1].split(':')
            ranking[0] = ["{0:02d}{1}{2:02d}".format(kind,timespan,market),
                          "{0}{1}{2}".format(tmp_date[0].zfill(4),tmp_date[1].zfill(2),tmp_date[2].zfill(2)),
                          "{0}{1}".format(tmp_time[0].zfill(2),tmp_time[1].zfill(2)) ]
            
        div_rankingTableWrapper = bsoup.find("div", {"class": "rankingTableWrapper"})
        rankingTable = div_rankingTableWrapper.find("tbody");
        for tr_rankingTabledata in rankingTable.find_all("tr"):
            tds = tr_rankingTabledata.find_all("td")
            rank = int(tds[0].string)
            if kind == 1 and timespan == 'd':
                #順位    コード    市場    名称    時刻    取引値    前日比    出来高    掲示板
                #print rank, ": ", tds[1].string,"; ", tds[2].string,"; ", tds[3].string,"; ", float(tds[4].string.lstrip('+').replace(',','')),"; ", float(tds[6].string.replace(',','')),"; ", float(tds[7].string.replace(',',''))
                ranking[rank] = [ tds[1].string, tds[2].string, tds[3].string, 
                                 tds[4].string, 
                                 float(tds[5].string.replace(',','')), 
                                 float(tds[6].find("span").string.lstrip('+')), 
                                 float(tds[7].string.lstrip('+').replace(',','')),
                                 float(tds[8].string.replace(',','')) ]
            elif kind == 1 and timespan == 'w':
                ranking[rank] = [ tds[1].string, tds[2].string, tds[3].string, 
                                 float(tds[4].string.lstrip('+').replace(',','')), 
                                 #float(tds[5].string.replace(',','')), 
                                 float(tds[6].string.replace(',','')), float(tds[7].string.replace(',','')) ]
            elif kind == 33:
                print (rank, ": ", tds[1].string,"; ", tds[2].string,"; ", tds[3].string,"; ", tds[4],"; ", tds[5],"; ", tds[6],"; ", tds[7] )
            else:
                print ('ranking kind error')
                return ranking
        #output
        target_page = target_page + 1
        
    return ranking
#    for key in ranking:
#        if ranking[key][1] != u'東証ETF':
#            print key,": ",ranking[key]

ranking = yahooFinanceRanking(kind=ranking_kind,timespan=ranking_timespan,market=ranking_market,lastpage=ranking_lastpage)
f_name = 'yfr' + ranking[0][0]+'-'+ranking[0][1]+ranking[0][2] + '.csv'
#yahooFinanceRanking(ranking, timespan='w',page=2)

f_encode = 'sjis'
if not os.path.exists(filepath+f_name):
    with open(f_name,'wb') as csv_file:
        for rank in ranking:
            #if rank == 0: continue
            rowlist = [ rank ] + ranking[rank]
            itemnum = len(rowlist)
            icount = 0
            for item in rowlist:
                if type(item) == int or type(item) == float:
                    itemstr = str(item)
                else:
                    itemstr = item.encode(f_encode)
    #            sys.stdout.write(itemstr)
                csv_file.write(itemstr)
                icount = icount + 1
                if icount < itemnum:
                    csv_file.write(',')
            csv_file.write('\n')
    csv_file.close()
else:
    print ('file ' + f_name + ' already exists.')
print ('done.')
#table_stocksTable = divtmp.find("table")
#divtmp = bsoup.find_all("div",{"class":"lineFi clearfix"})
#<div class="lineFi clearfix">
#<div class="stocksDtl clearFix">

#print table_stocksTable.find("th", class_ = "symbol").string
#print table_stocksTable.find_all("td", class_ = "stoksPrice")[1].string
#print table_stocksTable.find("td", class_ = "change").find_all("span")

#for div in divtmp:
#    print div
#for div in tables:
#    print div
#    for row in table.find_all("td",{"class":"stoksPrice"}):
#        if row.string :
#            price = row.string

#print price
# 2カラム目に日経平均を記録します
#rowlist.append(timestamp)
# csvに追記敷きます
#writer.writerow(csv_list)
# ファイル破損防止のために閉じます
#f.close()
