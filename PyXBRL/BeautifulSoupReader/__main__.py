#
# -*- coding: utf-8 -*-
'''
Created on 2017/12/24

@author: sin
'''
import sys
import glob
#import urllib.request
#import lxml
from bs4 import BeautifulSoup #, NavigableString
#import pandas as pd
import re

switch = { 'soupout': True }

def main():
    if __name__ == '__main__':
        pass
    params = get_params()
    params['code'] = params['basedir'].rstrip('/').split('/')[-1]
    #tse-qcedussm-77510-20181017418792-ixbrl.htm
    #Attachment/0101010-qcbs01-tse-qcedjpfr-68610-2018-09-20-01-2018-11-01-ixbrl.htm
    file_name_pattern = {'Summary' : params['basedir'].rstrip('/')+'/Summary/tse-*-ixbrl.htm', 
                'BalanceSheet': params['basedir'].rstrip('/')+'/Attachment/*-qcbs*-tse-*-ixbrl.htm',
                'ProfitLoss': params['basedir'].rstrip('/')+'/Attachment/*-qcpl*-tse-*-ixbrl.htm',
                'ComprehensiveIncome': params['basedir'].rstrip('/')+'/Attachment/*-qcci*-tse-*-ixbrl.htm',
                'CashFlow': params['basedir'].rstrip('/')+'/Attachment/*-qccf*-tse-*-ixbrl.htm', 
                }
    file_name = dict()
    for key in file_name_pattern :
        file_list = glob.glob(file_name_pattern[key])
        if len(file_list) != 1 or len(file_list[0]) == 0:
            print('no or more than one file(s) exist: ', file_list, file_name_pattern[key])
            continue
        file_name[key] = file_list[0]
    
    root = BeautifulSoup('','lxml')
    ix_dict = dict()    
    for key in file_name:
        xrblfile = open(file_name[key], 'r', encoding='utf-8') 
        soup = BeautifulSoup(xrblfile, 'lxml')
        xrblfile.close()
        if switch['soupout'] :
            f = open(key+'.txt', 'w', encoding='utf-8')
            f.write(soup.prettify())
            f.close()

        ix_dict[key] = get_ix_dict(soup)

        root.append(root.new_tag(key))
        node =  root.find(key)
        make_xbrl_subtree(ix_dict[key], node)

    xbrdb = BeautifulSoup('','lxml')
    scode = root.find('SecuritiesCode')['text']
    sname = root.find('CompanyName')['text']
    xbrdb.append(BeautifulSoup('','lxml').new_tag('Report', code=scode, stockname=sname))
    xbrdb.Report.append(root)
    
#    print(xbrdb.Report.prettify())
    node = xbrdb.find('Report', code=params['code'])
    table = dict()
    for each in node.find('BalanceSheet').find('CurrentQuarterInstant').contents:
        if each.get('text') != None:
            table[each.name] = [each.get('text'), each.get('scale', 0)]
    
    trans_dict = {'CashAndDeposits': '現金および預金', 
                  'NotesAndAccountsReceivableTrade': '受取手形及び売掛金',
                  'ElectronicallyRecordedMonetaryClaimsOperatingCA': '電子記録債権',
                  'MerchandiseAndFinishedGoods' : '商品及び製品',
                  'WorkInProcess' : '仕掛品', 
                  'RawMaterialsAndSupplies': '原材料及び貯蔵品', 
                  'OtherCA': 'その他流動資産',
                  'AllowanceForDoubtfulAccountsCA': '貸倒引当金', 
                  'CurrentAssets': '流動資産合計',
                  'BuildingsAndStructures': '建物及び構築物', 
                  'BuildingsAndStructuresNet': '建物及び構築物（純額）', 
                  'AccumulatedDepreciationBuildingsAndStructures': '建物及び構築物減価償却累計額',
                  'AccumulatedDepreciationMachineryEquipmentAndVehicles': '機械装置及び運搬具減価償却累計額',
                  'MachineryEquipmentAndVehicles': '機械装置及び運搬具',
                  'MachineryEquipmentAndVehiclesNet': '機械装置及び運搬具（純額）',
                  'Land': '土地',
                  'ConstructionInProgress': '建設仮勘定',
                  'OtherNetPPE': 'その他有形固定資産（純額）',
                  'AccumulatedDepreciationOtherPPE': 'その他有形固定資産減価償却累計額',
                  'PropertyPlantAndEquipment': '有形固定資産合計',
                  'OtherIA': 'その他無形固定資産',
                  'IntangibleAssets': '無形固定資産合計',
                  'GuaranteeDepositsIOA': '差入保証金',
                  'OtherIOA': 'その他投資ほか資産',
                  'AllowanceForDoubtfulAccountsIOAByGroup': '貸倒引当金',
                  'InvestmentsAndOtherAssets': '投資その他の資産合計',
                  'NoncurrentAssets': '固定資産合計',
                  'Assets': '資産合計',
                  'NotesAndAccountsPayableTrade': '支払手形及び買掛金',
                  'ElectronicallyRecordedObligationsOperatingCL': '電子記録債務',
                  'OtherCL': 'その他流動負債',
                  'CurrentLiabilities': '流動負債合計',
                  'LongTermLoansPayable': '長期借入金',
                  'ShortTermLoansPayable': '短期借入金',
                  'CurrentPortionOfLongTermLoansPayable': '１年内返済予定の長期借入金',
                  'IncomeTaxesPayable': '未払法人税',
                  'ProvisionForBonuses': '賞与引当金',
                  'OtherNCL': 'その他固定負債',
                  'NoncurrentLiabilities': '固定負債合計',
                  'Liabilities': '負債合計', 
                  'CapitalStock' :'株主資本合計',
                  'CapitalSurplus' :'資本剰余金',
                  'RetainedEarnings' :'利益剰余金',
                  'CapitalStock' :'株主資本合計',
                  'ShareholdersEquity': '株主資本合計',
                  'ValuationDifferenceOnAvailableForSaleSecurities': 'その他有価証券評価差額',
                  'ForeignCurrencyTranslationAdjustment': '為替換算調整勘定',
                  'ValuationAndTranslationAdjustments': 'その他の包括利益累計額',
                  'NetAssets': '純資産合計',
                  'LiabilitiesAndNetAssets': '負債純資産合計',
                  'NonControllingInterests': '非支配株主持分',
                  }
    for k in table:
        if k in trans_dict:
            print(trans_dict[k].rjust(14, '　') + '\t' + table[k][0].rjust(12, ' ') + ' ('+table[k][1]+')' )
        else:
            print(k.rjust(23, ' '), end='')
            print('\t' + table[k][0].rjust(12, ' ') + ' (' + table[k][1] + ')')
    exit()
    
    summary = get_summary(docroot)
        
    for key in summary: 
        print(key + ' = ' + str(summary[key]) )
#    print(curr2q.ConsolidatedMember.ResultMember.prettify())
    exit()

def make_xbrl_subtree(ix_dict, docroot):
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
                node.append(BeautifulSoup('','lxml').new_tag(tag_name))
                nextnode = node.find(tag_name)
            node = nextnode
        for tag_dict in ix_dict[a_key]:
            newtag = BeautifulSoup('','lxml').new_tag(tag_dict['name'])
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
                    if 'sign' in tag_dict and tag_dict['sign'] == '-':
                        tag_dict['text'] = '-' + tag_dict['text']
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
            ix_sign = node.get('sign')
            if ix_sign:
                ix_tag['sign'] = ix_sign
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
