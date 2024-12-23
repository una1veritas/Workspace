'''
Created on 2024/12/24

@author: sin
'''
import os, glob, sys
from datetime import datetime

dirpath = u'/Users/sin/Downloads/C11V101-2024-1100301601-レポート課題提出　受付-135551'

if __name__ == '__main__':
    if not os.path.isdir(dirpath) :
        print('error: this is not a directory path.')
        exit()
    table = dict()
    fdname_list = glob.glob(os.path.join(dirpath, '*.pdf'))
    assigntable = list()
    for each in fdname_list:
        if not os.path.isdir(each):
            assigntable.append(os.path.basename(each).split('_'))
            assigntable[-1].append(str(os.path.getsize(each)//1024))
    
    with open('output.csv', 'w') as wf:
        wf.write('Path = ' + dirpath + '\n')
        wf.write('Listed date and time = ' + datetime.today().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        for each in assigntable:
            wf.write(','.join( [each[0], each[1], each[-1]] ) + '\n')
            print(','.join( [each[0], each[1], each[-1]] ))
    
    print('file closed.')