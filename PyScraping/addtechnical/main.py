
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
header = []
for fname in files_list:
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # ヘッダーを読み飛ばしたい時
        if len(header) == 0 : 
            header = next(csv_reader)  # ヘッダーを読み飛ばしたい時
        else:
            next(csv_reader)
        for row in csv_reader:
            for index in range(1,len(row)):
                row[index] = int(row[index])
            tseries.append(row)

print (header)
tseries.sort()

#moving average
avrspans = [ 5, 21, 55 ]
for span in avrspans:
    for d in range(span - 1, len(tseries)) :
        psum = 0
        vsum = 0
        for i in range(d - span + 1, d) :
            psum = psum + tseries[i][4]*tseries[i][5]
            vsum = vsum + tseries[i][5]
        avr = round(psum / vsum, 1)
        tseries[d].append(avr)

for row in tseries:
    print(row)
#for code in ranking:
#    print(code,': ',ranking[code],'  ',codedict[code])

#    tbl = pandas.read_csv(file,skiprows=1,header=None)
#    tbl = tbl.drop(9,axis=1)
#    tbl.columns = ['rank', 'code', 'market', 'name', 'date', 'price', 'ratio', 'diff', 'volume' ]
#    if count == 1 :
#        ranking = tbl.ix[:,['code', 'rank']]
#        ranking = ranking.sort_values(by='code')

