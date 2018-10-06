# coding: UTF-8
import pandas
import datetime
import os
import glob
import sys
#from pymysql.constants.FIELD_TYPE import DATE
#from math import nan

args = dict()
args['base_path'] = 'C:/Users/Sin Shimozono/Downloads'
args['file_pattern'] = './20180914_Short_Positions.xls'
args['codes'] = []
index = 1
while index  < len(sys.argv) :
    if sys.argv[index][0] == '-' :
        if sys.argv[index] == '-path' :
            index = index + 1
            args['base_path'] = sys.argv[index]
        elif sys.argv[index] == '-code' :
            index = index + 1
            for a_code in sys.argv[index].split(':') :
                args['codes'].append(int(a_code))
        else:
            args[sys.argv[index]] = True
    else:
        args['file_pattern'] = sys.argv[index]
    index = index + 1

print(args)
os.chdir(args['base_path'])
file_list = sorted([os.path.abspath(path) for path in glob.glob(args['file_pattern'])])
#print(file_list)
#exit()

def xls_read(file_name, codes, charset='utf8'):
    column_names = ['lmargin',
                    'Date',
                    'Code',
                    'Name',
                    'Name_En',
                    'Short_Seller',
                    'Seller_Address',
                    'DIC',
                    'DIC_Address',
                    'Inv_Fund',
                    'Ratio',
                    'Number',
                    'Units',
                    'Prev_Date',
                    'Prev_Ratio',
                    'Notes',
                    'rmargin'
                    ]
    column_list=[1, 2,3,4, 5,6, 7,8, 9, 10,11,12, 15 ]
    data_types = {'Date' : datetime.date,
                'Code': str,
                'Name' : str,
                'Name_En' : str,
                'Short_Seller' : str,
                'Seller_Address':str,
                'DIC' : str,
                'DIC_Address' : str,
                'Inv_Fund' : str,
                'Ratio' : float,
                'Number' : int,
                'Units' : int,
                'Prev_Date' : datetime.date,
                'Prev_Ratio' : float,
                'Notes' : str,
                }
    df = pandas.read_excel(file_name,
    #                       skiprows=7,
                           usecols=column_list,
                           names=[column_names[col_index] for col_index in column_list],
                           dtype=data_types)
#    file_timestamp = datetime.datetime.today()
#    if str(df.iat[2,0])[:5] == u'公表年月日' :
#        file_timestamp = datetime.datetime.strptime(str(df.iat[2,1]), '%Y-%m-%d %H:%M:%S')
#     print('Date of disclosure: ' + file_timestamp.strftime('%Y-%m-%d'))
    df.drop(index=[0,1,2,3,4,5,6],inplace=True)
    if len(codes) != 0:
        df = df.loc[ df['Code'].isin(codes) ]
    if charset == 'sjis':
        # replace chars not compatible with shift-jis
        replace_dict = {'\u2013':'-', u'\uff0d': u'\u2212', u'\xa0': '', '\u3231': u'(株)' }
        for (t_key, t_value) in replace_dict.items() :
            df['Name'] = df['Name'].str.replace(t_key, t_value)
            df['Short Seller'] = df['Short Seller'].str.replace(t_key, t_value)
    #print(df)
    return df

df = pandas.DataFrame()
for filename in file_list:
    print(filename)
    df = df.append(xls_read(filename, args['codes']))
print('reading files finished.')

#df.sort_values(['Code','Short Seller', 'Date'],inplace=True)
#df = df[['Code', 'Short Seller', 'Date', 'Ratio']]

# short_sellers = sorted(df['Short Seller'].unique())
# for a_seller in short_sellers:
#     print(a_seller)
#     table = df.loc[(df['Short Seller'] == a_seller)][['Code','Name','Date','Ratio']]
#     table.rename(columns={'Ratio':a_seller})
#     print(table.rename(columns={'Ratio':a_seller}))
# exit()
df.set_index(['Date','Code','Short_Seller'], inplace=True)
df.sort_index(inplace=True)
df.to_csv('short-positions-'+'-'.join([str(t) for t in args['codes']])+'-'+datetime.datetime.today().strftime('%Y%m%d')+'.csv') #, encoding='shift-jis')
print('writing output finished.')