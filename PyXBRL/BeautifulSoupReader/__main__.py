#
# -*- coding: utf-8 -*-
'''
Created on 2017/12/24

@author: sin
'''
import sys
import urllib.request
#import xml.etree.ElementTree as ElementTree
import lxml
from bs4 import BeautifulSoup, NavigableString
import pandas as pd
import re

def main():
    if __name__ == '__main__':
        pass
    params = get_params()
    url_sum = params['baseurl'] + 'Summary/tse-qcedjpsm-78110-20181015378110-ixbrl.htm'
    url_bs = params['baseurl'] + 'Attachment/0101010-qcbs01-tse-qcedjpfr-78110-2018-08-31-01-2018-10-15-ixbrl.htm'
    html = urllib.request.urlopen(url_sum).read().decode('utf-8') 
    soup = BeautifulSoup(html, 'lxml')

    #        result = [[td.get_text(strip=True) for td in trs.select('th, td')] for trs in a_table.select('tr')]
    #df = pd.read_html(str(a_table), header=0, index_col=0)
    tables = list()
    curr_node = soup.body.find('div', class_='root').div
    for child in curr_node.children:
        if not isinstance(child, NavigableString): 
            for table_node in child.find_all('table'):
                if len(''.join([node.get_text(strip=True) for node in table_node.find_all('span')])) :
                    tables.append(get_table(child))
    #tds = [td.get_text(strip=True) for td in a_row.select('th, td')]
    
    for k in tables:
        print(k)
        print()
    '''
    xbrl = {}
    xbrl['head/title'] = soup.head.title.text

    basic_info(tables[0], xbrl)
    opresult_info(tables[4], tables[6], xbrl)
    for key in xbrl:
        print(key)
        print(xbrl[key])
        print()
    
    for key in xbrl['連結経営成績（累計）']:
        print(xbrl['連結経営成績（累計）'][key])
    print()
    ''' 
    exit()

def get_table(node):
    res_table = []
    for row in node.select('tr'):
        columns = []
        for td in row.select('th, td'):
            if '<ix:' in str(td.contents):
                print(re.findall('<ix:\w+[^>]+>', str(td.contents)))
            if td.span :
                columns.append(''.join([s.get_text(strip=True) for s in td.find_all('span')]))
            else:
                columns.append(td.get_text(strip=False))
        if sum([len(td) for td in columns]):
            res_table.append(columns)
    return res_table

def collect_tables(node):
    tables = node.find_all('table')
    for t_index in range(0, len(tables)):
        res_table = []
        for row in tables[t_index].select('tr'):
            columns = []
            for td in row.select('th, td'):
                if td.span :
                    columns.append(' '.join([s.get_text(strip=True) for s in td.find_all('span')]))
                else:
                    columns.append(td.get_text(strip=True))
            if sum([len(td) for td in columns]):
                res_table.append(columns)
        tables[t_index] = res_table
    return tables
    
def basic_info(tbl, resdict):
#    print(tbl)
    resdict['見出し'] = tbl[0][0]
    resdict['日付'] = tbl[1][2]
    if tbl[2][0] == '上場会社名':
        resdict['上場会社名'] = tbl[2][1]
        resdict['上場取引所'] = tbl[3][2]
    resdict['コード番号'] = tbl[3][1]
    resdict['四半期報告書提出予定日'] = tbl[6][1]
    resdict['配当支払開始予定日'] = tbl[6][3]
    return

def opresult_info(tbl, tbl2, resdict):
    result = dict()
    result['見出し'] = tbl[0][1:] + tbl2[0][1:]
    result[tbl[2][0]] = tbl[2][1:] + tbl2[2][1:]
    result[tbl[3][0]] = tbl[3][1:] + tbl2[3][1:]
    resdict['連結経営成績（累計）'] = result
    return
    
def get_params():
    params = { 'arg': [] }
    args = sys.argv[1:]
    while len(args) > 0:
        arg = args.pop(0)
        if arg[0] == '-' :
            if '=' in arg :
                (arg_key, arg_val) = arg[1:].split('=')
            else:
                arg_key = arg[1:]
                arg_val = args.pop(0)
            params[arg_key] = arg_val
        else:
            params['arg'].append(arg)
    return params

main()