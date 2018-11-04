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
    file_name = dict()
    #tse-qcedussm-77510-20181017418792-ixbrl.htm
    #Attachment/0101010-qcbs01-tse-qcedjpfr-68610-2018-09-20-01-2018-11-01-ixbrl.htm
    patterns = {'summary' : params['basedir'].rstrip('/')+'/Summary/tse-*-ixbrl.htm', 
                'bs': params['basedir'].rstrip('/')+'/Attachment/*-qcbs*-tse-*-ixbrl.htm',
                'pl': params['basedir'].rstrip('/')+'/Attachment/*-qcpl*-tse-*-ixbrl.htm',
                'ci': params['basedir'].rstrip('/')+'/Attachment/*-qcci*-tse-*-ixbrl.htm',
                'cf': params['basedir'].rstrip('/')+'/Attachment/*-qccf*-tse-*-ixbrl.htm', 
                }
    ix_dict = dict()
    for key in patterns :
        file_pattern = patterns[key]
        file_name[key] = glob.glob(file_pattern)
        if len(file_name[key]) != 1 or len(file_name[key][0]) == 0:
            print('no or more than one file(s) exist: ', file_name[key], file_pattern)
            continue
        xrblfile = open(file_name[key][0], 'r', encoding='utf-8') 
        soup = BeautifulSoup(xrblfile, 'lxml')
        xrblfile.close()
        f = open(key+'.txt', 'w', encoding='utf-8')
        f.write(soup.prettify())
        f.close()

        ix_dict[key] = get_ix_dict(soup)

    xbrl_info = BeautifulSoup('','lxml')
    xbrl_info.append(xbrl_info.new_tag('database'))
    xbrl_info.database.append(xbrl_info.new_tag('document'))
    xbrl_info.database.document.append(xbrl_info.new_tag('bs'))

    make_xbrl_subtree(ix_dict['bs'], xbrl_info.database.document.bs, xbrl_info)
    print(xbrl_info.database.document.bs.prettify())
    exit()
    
    summary = get_summary(docroot)
        
    for key in summary: 
        print(key + ' = ' + str(summary[key]) )
#    print(curr2q.ConsolidatedMember.ResultMember.prettify())
    exit()

def make_xbrl_subtree(ix_dict, docroot, bs4):
#     xbrldb = BeautifulSoup('','lxml')
#     xbrldb.append(xbrldb.new_tag('database'))
#     xbrldb.database.append(xbrldb.new_tag('document'))
    for a_key in sorted(ix_dict):
        if a_key == 'NoContextref' :
            continue
        node = docroot
        for tag_name in a_key.split('_'):
            nextnode = node.find(tag_name)
            if nextnode == None:
                node.append(bs4.new_tag(tag_name))
                nextnode = node.find(tag_name)
            node = nextnode
        for tag_dict in ix_dict[a_key]:
            newtag = bs4.new_tag(tag_dict['name'])
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
                    tag_dict['text'] = tag_dict['text'].replace(',','')
                    tag_dict.pop('format')
            for attr_name in tag_dict:
                if not (isinstance(tag_dict[attr_name], str) and tag_dict[attr_name].strip() == '') :
                    newtag[attr_name] = tag_dict[attr_name]
            node.append(newtag)
    
def get_ix_dict(bs):
    tag_pattern = re.compile('ix:\w*')
    contents = dict()
    for node in bs.body.find_all(tag_pattern):
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
            ix_tag['text'] = node.get_text(strip=True)

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
    return contents            
                
def get_summary(root_node):
    result = dict()
    patt = re.compile('CurrentAccumulatedQ[1234]Instant')
    instant_node = root_node.find(patt)
    result['DocumentName']   = instant_node.find('DocumentName').get('text')
    result['CompanyName']    = instant_node.find('CompanyName').get('text')
    result['SecuritiesCode'] = instant_node.find('SecuritiesCode').get('text')
    patt = re.compile('CurrentAccumulatedQ[1234]Duration')
    duration_node = root_node.find(patt)
    if 'IFRS' in result['DocumentName'] :
        result['Standards'] = 'IFRS'
    else:
        result['Standards'] = 'Japanese'
        
    japanese_entries = ['NetSales', 'ChangeInNetSales', 
                    'OperatingIncome', 'ChangeInOperatingIncome', 
                    'OrdinaryIncome', 'ChangeInOrdinaryIncome', 
                    'ProfitAttributableToOwnersOfParent', 'ChangeInProfitAttributableToOwnersOfParent', 
                    'ComprehensiveIncome', 'ChangeInComprehensiveIncome', 
                    'NetIncomePerShare', 'DilutedNetIncomePerShare', ]
    ifrs_entries = ['NetSalesIFRS', 'ChangeInNetSalesIFRS',
                          'OperatingIncomeIFRS', 'ChangeInOperatingIncomeIFRS',
                          'RevenueIFRS', 'ChangeInRevenueIFRS', 
                          'ProfitBeforeTaxIFRS', 'ChangeInProfitBeforeTaxIFRS',
                          'ProfitIFRS', 'ChangeInProfitIFRS', 
                          'ProfitAttributableToOwnersOfParentIFRS', 'ChangeInProfitAttributableToOwnersOfParentIFRS', 
                          'TotalComprehensiveIncomeIFRS', 'ChangesInTotalComprehensiveIncomeIFRS',
                          'BasicEarningsPerShareIFRS', 
                          'DilutedEarningsPerShareIFRS', 
                          ]
    if result['Standards'] == 'Japanese':
        entries = japanese_entries
    elif result['Standards'] == 'IFRS':
        entries = ifrs_entries
    for each in entries:
        each_entry = duration_node.find(each)
        if not each_entry:
            print('lacks: ',each)
            continue
        
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
        result[each] = val
    return result
    
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
