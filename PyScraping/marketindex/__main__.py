# coding: UTF-8
'''
Created on 2017/12/24

@author: sin
'''
import sys
import math
from pyquery import PyQuery
import pandas as pd
from sys import api_version

#import csv

if __name__ == '__main__':
    pass

url = 'https://kabutan.jp/'
pyquery = PyQuery(url)
for item in pyquery('div#wrapper_main')('table#header_shisuu_big') :
    for tr in item('table') :
        row = item('tr')
        if len(row('td')) != 0 :
            print(row('td').text())

#    for table in pyquery('div.padT12')('table'):
#        pytable = pyquery(table)
#        for tr in pytable('tr'):
#            row = pytable(tr)
#            #skip header line
#            if len(row('td')) == 0: 
#                continue
#            columns = [code]
#            colnum = 0
#            for td in row('td'):
#                td_str = row(td).text()
#                if colnum == 0 : 
#                    td_date = td_str.replace(u'年','/').replace(u'月','/').replace(u'日','').split('/')
#                    td_str = str(td_date[0]).zfill(4)+'/'+str(td_date[1]).zfill(2)+'/'+str(td_date[2]).zfill(2)
