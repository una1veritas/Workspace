
import glob
import csv
import sys

if len(sys.argv) < 2:
    exit
stockcode = sys.argv[1]
print (stockcode)
files_list = glob.glob('../yfseries/'+stockcode+'-*-*.csv')
print(files_list)

tseries = []
for fname in files_list:
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # ヘッダーを読み飛ばしたい時
        header = next(csv_reader)  # ヘッダーを読み飛ばしたい時
        for row in csv_reader:
            for index in range(1,len(row)):
                row[index] = int(row[index])
            tseries.append(row)

tseries[1:].sort(reverse=False)

averages = []
avrspans = [ 5, 21, 55 ]

for dnum in range(0,len(tseries)):
    avrrow = [ tseries[dnum][0] ]
    for span in avrspans:
        if dnum > span:
            avr = 0
            for d in range(dnum-span+1,dnum):
                avr = avr + tseries[d][4]
            avr = avr/span
            avrrow.append(avr)
    averages.append(avrrow)

print(averages)
#for code in ranking:
#    print(code,': ',ranking[code],'  ',codedict[code])

#    tbl = pandas.read_csv(file,skiprows=1,header=None)
#    tbl = tbl.drop(9,axis=1)
#    tbl.columns = ['rank', 'code', 'market', 'name', 'date', 'price', 'ratio', 'diff', 'volume' ]
#    if count == 1 :
#        ranking = tbl.ix[:,['code', 'rank']]
#        ranking = ranking.sort_values(by='code')

