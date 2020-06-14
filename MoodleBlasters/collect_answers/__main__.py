'''
Created on 2020/05/21

@author: sin
'''
# coding: utf-8
import sys
import os
import re
from bs4 import BeautifulSoup

if len(sys.argv) > 1:
    basedir = sys.argv[1]
else:
    print('usage: command dir-path', file=sys.stderr)
    exit()

if not os.path.exists(basedir+'/namelist.txt') :
    print('namelist.txt not exists.')
    exit(1)
else:
    namelist = dict()
    with open(basedir+'/namelist.txt', encoding='utf-8') as txtf:
        for a_line in txtf:
            a_row = a_line.split(',')
            if '\ufeff' in a_row[0] :
                continue
            namelist[a_row[0]] = (a_row[0], a_row[1],a_row[3],a_row[5]) 

workdb = dict()
workdb['date'] = dict()
for entry in namelist :
    workdb[entry] = dict()

dirlist = [basedir + '/' + path for path in os.listdir(basedir) if os.path.isdir(basedir + '/' + path)]
for probfolder in dirlist:
    probprop = probfolder.replace('¥','/').split('/')[-1].split('_')
    if probprop[0] != '出席確認課題' :
        continue
    print("entry: " + str(probprop) )
    if probprop[3] not in workdb['date'] :
        workdb['date'][probprop[3]] = probprop[1]+'-'+probprop[2]
    if os.path.isdir(probfolder):
        indivs = [ folder for folder in os.listdir(probfolder)]
        for entry in indivs: 
            oltextpath = probfolder + '/' + entry + '/' + 'onlinetext.html'
            if os.path.exists(oltextpath) :
                submprop = entry.split('_')
                with open(oltextpath, encoding='utf-8') as txtf:
                    contstr = txtf.read()
                #if len(contstr):
                bsoup = BeautifulSoup(contstr,"html.parser")
                contstr = bsoup.get_text()
                contstr = contstr.replace('\u3000', '')
                contstr = contstr.replace('\xa0', ' ')
                contstr = contstr.replace('\r', ' ')
                contstr = contstr.replace('\n', ' ')
                contstr = contstr.replace('出席確認課題', ' ')
                contstr = contstr.replace('出席確認', ' ')
                contstr = contstr.replace('演習問題', '')
                contstr = contstr.replace('演習課題', '')
                contstr = contstr.replace('学生番号：182C1073', '')
                contstr = contstr.replace('名前：黒木冬悟', '')
                contstr = contstr.replace('192C1177', '')
                contstr = contstr.replace('吉村唯吹', '')
                contstr = contstr.replace('吉村唯', '')
                contstr = contstr.replace('192C1005 有吉優聖','')
                contstr = contstr.replace('192C1005有吉優聖','')
                contstr = contstr.strip()[:16]
                if len(contstr) :
                    workdb[submprop[0]][probprop[3]] = contstr
                else:
                    print('EMPTY!!')
            else:
                print('not exist: '+oltextpath)

table = [['学生番号', '氏名']]
dates = list()
for date in sorted(workdb['date'].keys()) :
    table[0].append(date[:2]+'/'+date[2:])
    dates.append(date)
for sid in sorted(workdb.keys()):
    if sid == 'date' :
        continue
    row = [sid, namelist[sid][1]]
    row = row + [workdb[sid][a_date] if a_date in workdb[sid] else u'(欠席)' for a_date in dates]
    table.append(row)

with open(basedir+'/table.txt', mode='w') as wf:
    for each in table:
        for an_item in each:
            wf.write(an_item)
            wf.write('\t')
        wf.write('\n')

print(workdb['192C1005'])
print('finished.')
