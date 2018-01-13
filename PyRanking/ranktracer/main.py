
import glob
import pandas

file_list = glob.glob('../yfranking/yfr1d2a-*.csv')
sorted(file_list,reverse=True)
print(file_list)

dtable = pandas.read_csv('../yfranking/yfr1d2a-201801121740.csv',skiprows=1, header=None)
dtable = dtable.drop(9, axis=1)
print(dtable.columns) # = ['rank', 'code', 'market', 'name', 'date', 'price', 'ratio', 'diff', 'volume' ]
dtable.columns = ['rank', 'code', 'market', 'name', 'date', 'price', 'ratio', 'diff', 'volume' ]

ranking = dtable.ix[:,['rank','code']]
    
print(ranking.info())