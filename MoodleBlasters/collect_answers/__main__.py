'''
Created on 2020/05/21

@author: sin
'''

import sys
import os
from bs4 import BeautifulSoup

if len(sys.argv) > 1:
    basedir = sys.argv[1]
else:
    print('usage: command dir-path', file=sys.stderr)
    exit()
    
dirlist = [basedir + '/' + path for path in os.listdir(basedir)]
for probfolder in dirlist:
    print("entry: " + probfolder)
    if os.path.isdir(probfolder):
        indivs = [ folder for folder in os.listdir(probfolder)]
        for entry in indivs: 
            oltextpath = probfolder + '/' + entry + '/' + 'onlinetext.html'
            if os.path.exists(oltextpath) :
                with open(oltextpath, encoding='utf-8') as txtf:
                    contstr = txtf.read()
                if len(contstr) :
                    bsoup = BeautifulSoup(contstr,"html.parser")
                    contstr = bsoup.get_text().replace('\n','')
                    contstr = contstr.replace('問題２', '\t')
                    contstr = contstr.replace('問題2', '\t')
                    contstr = contstr.split('\t')[0]
                    print(contstr)
                else:
                    print('EMPTY!!')
            else:
                print('not exist: '+oltextpath)
