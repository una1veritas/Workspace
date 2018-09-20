# coding: UTF-8
import pandas
import datetime
import os
import glob
import sys

args = dict()
args['base_path'] = 'C:/Users/Sin Shimozono/Downloads'
args['file_pattern'] = './*_Short_Positions.xls'
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

def xls_read(file_name, codes):
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
    column_list=[1,2,3, 5,  10,11, ]
    data_types = {#'Date' : str,
                'Code': int,
                'Name' : str,
    #            'Name (En)' : str,
                'Short Seller' : str,
    #            'Seller Address':str,
    #            'DIC' : str,
    #            'Address of DIC' : str,
    #            'Inv. Fund' : str,
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
    df = df.loc[ df['Code'].isin(codes) ]
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

#df.set_index(['Code', 'Short Seller', 'Date'], inplace=True)
df.sort_values('Date',inplace=True)
short_sellers = df['Short Seller'].unique()
print(sorted(short_sellers))
exit()
df.to_csv('short-positions-'+datetime.datetime.today().strftime('%Y%m%d')+'.csv', encoding='shift-jis')
print('writing output finished.')