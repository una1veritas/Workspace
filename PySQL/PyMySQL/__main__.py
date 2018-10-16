# coding: UTF-8
import sys
from urllib.parse import urlparse
import mysql.connector

def fetch_table(conection, query):
    cur = conection.cursor(dictionary=False)
    cur.execute(query)
    t = []
    while True:
        r = cur.fetchone()
        if r != None :
            t.append(r)
        else:
            break
    return t

def main():
    
    args = { 'code': 4666, 'date_from': '20171001'}
    for arg in sys.argv[1:]:
        if arg[:1] == '-':
            arg_name, arg_val = arg[1:].split('=')
            if len(arg_val) == 0 :
                arg_val = True
            if arg_name == 'code':
                arg_val = int(arg_val)
            args[arg_name] = arg_val
    print(args)
    
    url = urlparse('mysql://sin:qpm4nz@localhost:3306/stockanal')
    
    conn = mysql.connector.connect(
        host = url.hostname or 'localhost',
        port = url.port or 3306,
        user = url.username or 'root',
        password = url.password or '',
        database = url.path[1:],
    )
    
    if not conn.is_connected():
        print('connection is not available.')
        exit()
    
    # collect the names of sellers
    sql_sellers = 'SELECT DISTINCT Short_Seller FROM short_positions WHERE Date >= \'{date}\' AND Code = {code} ORDER BY Short_Seller'
    sql_sellers = sql_sellers.format(date=args['date_from'], code=args['code'])
    seller_list = [row[0] for row in fetch_table(conn, sql_sellers)]
    
    sql_name = 'SELECT DISTINCT Name FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'
    sql_name = sql_name.format(date=args['date_from'], code=args['code'])
    stockname_list = [row[0] for row in fetch_table(conn, sql_name)]
    
    short_positions = dict()
    sql = 'SELECT Date, Short_Seller, Number FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'
    for seller in seller_list:
        sql = sql.format(date=args['date_from'], code=args['code'], seller=seller)
        for row in fetch_table(conn,sql):
            if not row[0] in short_positions:
                short_positions[row[0]] = dict()
            short_positions[row[0]][row[1]] = row[2]
    
    sql = 'SELECT date, code, close FROM stock_price WHERE Date >= \'{date}\' AND Code = {code}'.format(date=args['date_from'], code=args['code'])
    sql = sql.format(date=args['date_from'], code=args['code'])
    for row in fetch_table(conn,sql):
        if not row[0] in short_positions:
            short_positions[row[0]] = dict()
        short_positions[row[0]][row[1]] = row[2]
    
    for col in ['date', args['code']] + seller_list:
        print(str(col)+'\t', end='')
    print()
    for eachday in sorted(short_positions.keys()):
        print(str(eachday), end='')
        for item in [args['code']] + seller_list:
            if item in short_positions[eachday]:
                print('\t'+str(short_positions[eachday][item]), end='')
            else:
                print('\t', end='')
        print()
        
    print('finished.')

main()