#
# -*- coding: utf-8 -*-
'''
Created on 2017/12/24

@author: sin
'''
import sys
import glob
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
    
    file_sum_name = params['basedir'].rstrip('/')+'/Summary/tse-qcedjpsm-*-ixbrl.htm'
    file_sum = glob.glob(file_sum_name)
    if len(file_sum) > 1 or len(file_sum) == 0:
        print('no or more than one file(s) exist: '+params['basedir']+'/Summary/tse-qcedjpsm*-ixbrl.thm', file_sum_name)
        exit()
    #html = urllib.request.urlopen(url_sum).read().decode('utf-8') 
    xrblfile = open(file_sum[0], 'r', encoding='utf-8') 
    soup = BeautifulSoup(xrblfile, 'lxml')
    xrblfile.close()

#     f = open('prettify.txt', 'w', encoding='utf-8')
#     f.write(soup.prettify())
#     f.close()
    ''' replace im-parsable ix:xxx tag to ix_xxx '''
#    source = str(soup)
#    source = source.replace('ix:','ix_')
#    soup = BeautifulSoup(source, 'lxml')
    
    #        result = [[td.get_text(strip=True) for td in trs.select('th, td')] for trs in a_table.select('tr')]
    #df = pd.read_html(str(a_table), header=0, index_col=0)
    curr_node = soup.body.find('div', class_='root')
    tag_pattern = re.compile('ix:\w*')
    for node in curr_node.find_all(tag_pattern):
        ix_type = node.name
        ix_contextref = node.get('contextref')
        ix_name = node.get('name')
        if ix_name:
            ix_name = ix_name.split(':')[1]
        ix_format = node.get('format')
        if ix_format:
            ix_format = ix_format.split(':')[1]
        print(ix_type, ix_contextref, ix_name, ix_format)
        
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

def get_ix_table(node):
    result_table = []
    result_dict = dict()
    for row in node.select('tr'):
        columns = []
        for td in row.select('th, td'):
#             ix_info = ''
#             if td.ix_nonnumeric != None:
#                 ix_info = ' '+' '.join([td.ix_nonnumeric.get(attr) for attr in ['name', 'format'] if td.ix_nonnumeric.get(attr) != None ])
#             elif td.ix_nonfraction != None:
#                 ix_info = ' '+' '.join([td.ix_nonfraction.get(attr) for attr in ['name', 'format'] if td.ix_nonfraction.get(attr) != None])
            if td.span :
                text = td.span.get_text(strip=True) #''.join([s.get_text(strip=True) for s in td.find_all('span')])
            else:
                text = td.get_text(strip=True)
#             if len(text) :
#                 text = text + ' ' +ix_info
            columns.append( text )
        if sum([len(td) for td in columns]):
            result_table.append(columns)
    return (result_table, result_dict)

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