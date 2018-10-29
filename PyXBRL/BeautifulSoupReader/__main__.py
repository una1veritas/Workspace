#
# -*- coding: utf-8 -*-
'''
Created on 2017/12/24

@author: sin
'''
import sys
import glob
#import urllib.request
import lxml
from bs4 import BeautifulSoup, NavigableString
#import pandas as pd
import re

def main():
    if __name__ == '__main__':
        pass
    params = get_params()
    #tse-qcedussm-77510-20181017418792-ixbrl.htm
    file_sum_name = params['basedir'].rstrip('/')+'/Summary/tse-*-ixbrl.htm'
    file_sum = glob.glob(file_sum_name)
    if len(file_sum) > 1 or len(file_sum) == 0:
        print('no or more than one file(s) exist: ', file_sum_name, file_sum)
        exit()
    #html = urllib.request.urlopen(url_sum).read().decode('utf-8') 
    xrblfile = open(file_sum[0], 'r', encoding='utf-8') 
    soup = BeautifulSoup(xrblfile, 'lxml')
    xrblfile.close()

    f = open('prettify.txt', 'w', encoding='utf-8')
    f.write(soup.prettify())
    f.close()
    ''' replace im-parsable ix:xxx tag to ix_xxx '''
#    source = str(soup)
#    source = source.replace('ix:','ix_')
#    soup = BeautifulSoup(source, 'lxml')
    
    #        result = [[td.get_text(strip=True) for td in trs.select('th, td')] for trs in a_table.select('tr')]
    #df = pd.read_html(str(a_table), header=0, index_col=0)
    curr_node = soup.body #.find('div', class_='root')
    tag_pattern = re.compile('ix:\w*')
    contents = dict()
    for node in curr_node.find_all(tag_pattern):
        ix_tag = dict()
        ix_tag['type'] = node.name.split(':')[1]
        ix_contextref = node.get('contextref')
        if ix_contextref :
            ix_tag['contextref'] = ix_contextref
        ix_name = node.get('name')
        if ix_name :
            ix_tag['name'] = ix_name.split(':')[1]
        ix_format = node.get('format')
        if ix_format:
            ix_tag['format'] = ix_format.split(':')[1]
            
#        if ix_tag['name'] in ['DocumentName', 'FilingDate', 'CompanyName', 'SecuritiesCode' ] :
        if ix_tag['type'] == 'nonnumeric' :
            ix_text = node.get_text(strip=True)
            if not ix_text:
                ix_text = [sp.get_text(strip=True) for sp in node.find_all('span')]
            if ix_text:
                ix_tag['text'] = ix_text
        if ix_tag['type'] == 'nonfraction':
            ix_scale = node.get('scale')
            if ix_scale:
                ix_tag['scale'] = ix_scale
            ix_tag['text'] = node.get_text(strip=True)

        if 'contextref' in ix_tag :
            ref = ix_tag.pop('contextref', None)
            if ix_tag['type'] == 'nonfraction':
                ix_tuple = (ix_tag['name'], ix_tag.get('format', ''), ix_tag.get('scale', ''), ix_tag.get('text', '') )
            elif ix_tag['type'] == 'nonnumeric':
                ix_tuple = (ix_tag['name'], ix_tag.get('format', ''), ix_tag.get('text', '') )
            else:
                ix_tuple = tuple(ix_tag)

            if not ref in contents:
                contents[ref] = [ ix_tuple ]
            else:
                contents[ref].append(ix_tuple)
        
    '''
    xbrl = {}
    xbrl['head/title'] = soup.head.title.text
    '''
    root = BeautifulSoup('','lxml')
    root.append(root.new_tag('DocRoot'))
    for a_key in sorted(contents):
        path = a_key.split('_')
        node = root.DocRoot
        for tag in path:
            if not node.find(tag) :
                node.append(root.new_tag(tag))
            node = node.find(tag)
        for each_tuple in contents[a_key]:
            if len(each_tuple) == 3 :
                new_tag = root.new_tag(each_tuple[0], format=each_tuple[1])
                if each_tuple[2]:
                    new_tag = root.new_tag(each_tuple[0], format=each_tuple[1], text=each_tuple[2])
                else:
                    new_tag = root.new_tag(each_tuple[0], format=each_tuple[1])
            else:
                if each_tuple[1] == "numdotdecimal" :
                    value = each_tuple[3].replace(',', '')
                    new_tag = root.new_tag(each_tuple[0], format=each_tuple[1], scale=each_tuple[2], text=value)
                else:
                    new_tag = root.new_tag(each_tuple[0], format=each_tuple[1], scale=each_tuple[2])
        node.append(new_tag)
    print(root.prettify())
    exit()

'''
def get_table(node):
    table = list()
    attrs = dict()
    for row in node.select('tr'):
        columns = list()
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
'''

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
