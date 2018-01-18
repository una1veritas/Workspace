
import glob
import csv
import sys

rankingname = '1d2a'
if len(sys.argv) > 1 : rankingname = sys.argv[1]
files_list = glob.glob('../yfranking/yfr'+rankingname+'-*.csv')
files_list.sort(reverse=False)
print(files_list)

ranking = { }
codedict = { }
file_name = files_list[-1]
print('initialize stock codes by the latest ranking in file ',file_name)
with open(file_name, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # ヘッダーを読み飛ばしたい時
    for row in csv_reader:
        ranking[row[1]] = [ ]
        codedict[row[1]] = [row[2], row[3], float(row[6].replace('%',''))]

print('slice')
pcount = 1
for file_name in files_list[-3:]:
    with open(file_name, 'r') as file:
        print('reading ', file_name)
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            if row[1] in ranking:
#                print(row[1],', ',row[0])
                ranking[row[1]].append(int(row[0]))
#                print(ranking[row[1]])
    for code in ranking:
        if len(ranking[code]) < pcount:
            ranking[code].append('NA')
    pcount = pcount + 1
    
print('histories of ranking:')
for code in ranking:
    print(code,': ',ranking[code],'  ',codedict[code])

#    tbl = pandas.read_csv(file,skiprows=1,header=None)
#    tbl = tbl.drop(9,axis=1)
#    tbl.columns = ['rank', 'code', 'market', 'name', 'date', 'price', 'ratio', 'diff', 'volume' ]
#    if count == 1 :
#        ranking = tbl.ix[:,['code', 'rank']]
#        ranking = ranking.sort_values(by='code')

