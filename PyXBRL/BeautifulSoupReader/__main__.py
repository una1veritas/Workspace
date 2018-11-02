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
    
    tag_pattern = re.compile('ix:\w*')
    contents = dict()
    for node in soup.body.find_all(tag_pattern):
        ix_tag = dict()
        ix_tag['type'] = node.name.split(':')[1]
        ix_tag['contextref'] = node.get('contextref', 'NoContextref')            
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
        elif ix_tag['type'] == 'nonfraction':
            ix_scale = node.get('scale')
            if ix_scale:
                ix_tag['scale'] = ix_scale
            ix_tag['value'] = node.get_text(strip=True)

        if 'contextref' in ix_tag :
            ref = ix_tag.pop('contextref', None)
#             if ix_tag['type'] == 'nonfraction':
#                 ix_tuple = (ix_tag['name'], ix_tag.get('format', ''), ix_tag.get('scale', ''), ix_tag.get('text', '') )
#             elif ix_tag['type'] == 'nonnumeric':
#                 ix_tuple = (ix_tag['name'], ix_tag.get('format', ''), ix_tag.get('text', '') )
#             else:
#                 ix_tuple = tuple(ix_tag)
#
            if not ref in contents:
                contents[ref] = [ ix_tag ]
            else:
                contents[ref].append(ix_tag)

    xbrldb = BeautifulSoup('','lxml')
    xbrldb.append(xbrldb.new_tag('database'))
    xbrldb.database.append(xbrldb.new_tag('document'))
    docroot = xbrldb.database.document
    
    for a_key in sorted(contents):
        if a_key == 'NoContextref' :
            continue
        node = docroot
        for tag_name in a_key.split('_'):
            nextnode = node.find(tag_name)
            if nextnode == None:
                node.append(xbrldb.new_tag(tag_name))
                nextnode = node.find(tag_name)
            node = nextnode
        for tag_dict in contents[a_key]:
            newtag = xbrldb.new_tag(tag_dict['name'])
            tag_dict.pop('name')
            tag_dict.pop('type')
            if 'format' in tag_dict: 
                if (tag_dict['format'] == 'booleantrue') or (tag_dict['format'] == 'booleanfalse'):
                    if tag_dict['format'] == 'booleantrue':
                        tag_dict['boolean'] = True
                    elif tag_dict['format'] == 'booleanfalse' :
                        newtag['boolean'] = False
                    tag_dict.pop('format')
                elif tag_dict['format'] == 'numdotdecimal':
                    tag_dict['value'] = tag_dict['value'].replace(',','')
                    tag_dict.pop('format')
            for attr_name in tag_dict:
                if not (isinstance(tag_dict[attr_name], str) and tag_dict[attr_name].strip() == '') :
                    newtag[attr_name] = tag_dict[attr_name]
            node.append(newtag)
    
    q_pattern = re.compile('CurrentAccumulatedQ[1234]Duration')
    print(docroot.prettify())
    curr_q2 = docroot.find(q_pattern)
    curr_q2_result_entry = ['NetSales', 'ChangeInNetSales', 
                    'OperatingIncome', 'ChangeInOperatingIncome', 
                    'OrdinaryIncome', 'ChangeInOrdinaryIncome', 
                    'ProfitAttributableToOwnersOfParent', 'ChangeInProfitAttributableToOwnersOfParent', 
                    'ComprehensiveIncome', 'ChangeInComprehensiveIncome', 
                    'NetIncomePerShare', 'DilutedNetIncomePerShare', ]
    '''
       <RevenueIFRS scale="6" value="884044">
   </RevenueIFRS>
   <ChangeInRevenueIFRS scale="-2" value="20.3">
   </ChangeInRevenueIFRS>
   <ProfitBeforeTaxIFRS scale="6" value="37915">
   </ProfitBeforeTaxIFRS>
   <ChangeInProfitBeforeTaxIFRS scale="-2" value="94.3">
   </ChangeInProfitBeforeTaxIFRS>
   <ProfitIFRS scale="6" value="30272">
   </ProfitIFRS>
   <ChangeInProfitIFRS scale="-2" value="80.6">
   </ChangeInProfitIFRS>
   <ProfitAttributableToOwnersOfParentIFRS scale="6" value="27241">
   </ProfitAttributableToOwnersOfParentIFRS>
   <ChangeInProfitAttributableToOwnersOfParentIFRS scale="-2" value="77.4">
   </ChangeInProfitAttributableToOwnersOfParentIFRS>
   <TotalComprehensiveIncomeIFRS scale="6" value="36764">
   </TotalComprehensiveIncomeIFRS>
   <ChangesInTotalComprehensiveIncomeIFRS>
   </ChangesInTotalComprehensiveIncomeIFRS>
   <BasicEarningsPerShareIFRS scale="0" value="21.78">
   </BasicEarningsPerShareIFRS>
   <DilutedEarningsPerShareIFRS scale="0" value="21.77">
   </DilutedEarningsPerShareIFRS>
    '''
    curr_q2_result = dict()
    for each in curr_q2_result_entry:
        each_entry = curr_q2.find(each)
        if each_entry.get('scale') :
            scale = int(each_entry.get('scale')) 
        else:
            scale = 0
        if each_entry.get('value') :
#            print(each_entry.get('value'))
            val = float(each_entry.get('value'))
        else:
            val = 0
        if scale != 0 :
            val = val * (10**scale)
        curr_q2_result[each] = val
    
    curr_q2_result['DocumentName'] = docroot.CurrentAccumulatedQ2Instant.find('DocumentName').get('text')
    curr_q2_result['CompanyName'] = docroot.CurrentAccumulatedQ2Instant.find('CompanyName').get('text')
    curr_q2_result['SecuritiesCode'] = docroot.CurrentAccumulatedQ2Instant.find('SecuritiesCode').get('text')
    
    for key in curr_q2_result: 
        print(key + ' = ' + str(curr_q2_result[key]) )
#    print(curr2q.ConsolidatedMember.ResultMember.prettify())
    exit()

# def get_table(node):
#     table = list()
#     attrs = dict()
#     for row in node.select('tr'):
#         columns = list()
#         for td in row.select('th, td'):
#             if td.span :
#                 text = td.span.get_text(strip=True) #''.join([s.get_text(strip=True) for s in td.find_all('span')])
#             else:
#                 text = td.get_text(strip=True)
#             columns.append( text )
#         if sum([len(td) for td in columns]):
#             result_table.append(columns)
#     return (result_table, result_dict)
    
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
