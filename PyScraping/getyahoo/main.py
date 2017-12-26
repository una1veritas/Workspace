# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import urllib2
from bs4 import BeautifulSoup
from datetime import datetime
import sys
#import csv
#import time

if __name__ == '__main__':
    pass

scode = "998407.O"
if len(sys.argv) > 1:
    scode = sys.argv[1]
    print scode
#rowlist = []

timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
# 1カラム目に時間を挿入します
#rowlist.append(timestamp)

url = "https://stocks.finance.yahoo.co.jp/stocks/detail/?code=" + scode
htmlsrc = urllib2.urlopen(url)

bsoup = BeautifulSoup(htmlsrc, "html.parser")
divtmp = bsoup.find("div", class_="stocksDtl clearFix")
table_stocksTable = divtmp.find("table")
divtmp = bsoup.find_all("div",{"class":"lineFi clearfix"})
#<div class="lineFi clearfix">
#<div class="stocksDtl clearFix">

print table_stocksTable.find("th", class_ = "symbol").string
print table_stocksTable.find_all("td", class_ = "stoksPrice")[1].string
print table_stocksTable.find("td", class_ = "change").find_all("span")

for div in divtmp:
    print div
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
