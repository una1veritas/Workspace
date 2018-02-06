
import glob
import csv
import sys
from operator import itemgetter

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
    with open(file_name, 'r') as file:
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
            rankhist[row[1]][mondate] = round(float(row[6])/(float(row[4]) - float(row[6])),3)
#print(ranking)
print(dates)
#print(codedict)

riseup = [ ]
for code in rankhist:
    riseup.append([code, len(rankhist[code]), list(rankhist[code].values())])

riseup = sorted(riseup, key=itemgetter(1), reverse=True)
for row in riseup:
    if row[1] < len(dates)/2 :
        break
    print(row[0],',',row[1],',',row[2])