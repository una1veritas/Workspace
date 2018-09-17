# coding: UTF-8
import pandas
import datetime
import os
import glob
import sys


base_path = 'C:/Users/Sin Shimozono/Downloads'
print(sys.argv)
base_path=sys.argv[1]
os.chdir(base_path)
fileabspath_list = [os.path.abspath(path) for path in glob.glob('./*_Short_Positions.xls')]

xls_name=fileabspath_list[0] #'C:/Users/Sin Shimozono/Documents/Workspace/Python/20180723_Short_Positions.xls'
column_names = ['leftmargin',
                'Date',
                'Code',
                'Name',
                'Name (En)',
                'Short Seller',
                'Address of Seller',
                'DIC',
                'Address of DIC',
                'Inv. Fund',
                'Ratio',
                'Number',
                'Units',
                'Prev. Date',
                'Prev. Ratio',
                'Notes',
                'rightmargin'
                ]
column_list=[1,2,3, 5,  10,11, 13,14 ]
data_types = {#'Date' : str,
            'Code': int,
            'Name' : str,
#            'Name (En)' : str,
            'Short Seller' : str,
#            'Address of Seller':str,
#            'DIC' : str,
#            'Address of DIC' : str,
#            'Inv. Fund' : str,
            'Notes' : str,
            }
df = pandas.read_excel(xls_name,
#                       skiprows=7,
                       usecols=column_list,
                       names=[column_names[col_index] for col_index in column_list],
                       dtype=data_types)

dofd = datetime.datetime.today()
if str(df.iat[2,0])[:5] == '公表年月日' :
    dofd = datetime.datetime.strptime(str(df.iat[2,1]), '%Y-%m-%d %H:%M:%S')
print('Date of disclosure: ' + dofd.strftime('%Y-%m-%d'))

df.drop(index=[0,1,2,3,4,5,6],inplace=True)
df = df.set_index(['Code', 'Name', 'Short Seller', 'Date'], drop=True)
#print(df)

df.to_csv('short-selling-'+dofd.strftime('%Y%m%d')+'.csv', encoding='shift-jis')
