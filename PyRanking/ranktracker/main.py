
import glob
import csv
import sys
from operator import itemgetter
import numpy as np

basedirname = './data'
rankingname = '1d2a'
if len(sys.argv) > 2 : 
    basedirname = sys.argv[1]
    rankingname = sys.argv[2]
    backto = int(sys.argv[3])
files_list = glob.glob(basedirname+'/'+'yfr'+rankingname+'-*.csv')
files_list.sort(reverse=False)
print(files_list[-backto:])

# collect codes 
rankhist = { }
codedict = { }
dates = [ ]
for file_name in files_list[-backto:] :
    with open(file_name, 'r', encoding='utf-8') as file:
        header = file.readline().split('\n')[0]
        [dstr, tstr] = header.split(',')[2:4]
        thisdate = (int(dstr[0:4]), int(dstr[4:6]), int(dstr[6:8]), int(tstr[0:2]), int(tstr[2:4]))
        dates.append(thisdate)
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not row[1] in rankhist:
                codedict[row[1]] = ( row[2], row[3] )
                rankhist[row[1]] = { }
            mondate = thisdate[0:3]
            rankhist[row[1]][mondate] = [ round(float(row[6])/(float(row[4]) - float(row[6])),3), float(row[4]) ]
#print(ranking)
print(dates)
#print(codedict)

riseup = [ ]
for code in rankhist:
    histlist = [ ]
    for d in dates:
        #print(d[0:3], rankhist[code])
        if d[0:3] in rankhist[code] :
            dateint = (d[0]%100)*10000+d[1]*100+d[2]
            histlist.append([ dateint ] + rankhist[code][d[0:3]])
    histlist.reverse()
    riseup.append([code, len(rankhist[code]), histlist ])

riseup = sorted(riseup, key=itemgetter(1,2), reverse=True)
for row in riseup:
    if row[1] < len(dates)/2 :
        break
    x = []
    y = []
    for ix in range(len(row[2])) :
        x.append(len(row[2]) - ix - 1)
        y.append(row[2][ix][2])
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    a, c = np.linalg.lstsq(A,y,rcond=None)[0]
    print(str(row[0])+ ' ' + codedict[row[0]][1])
    print('\t'+str(row[1]) + ' ' + ' ('+str(round(a,1))+') ', end='')
    for t in row[2]:
        print('('+str(t[0])+' '+str(t[2])+')'+', ', end='')
    print()
    
print('done.')