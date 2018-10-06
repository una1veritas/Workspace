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

args = { 'code': 5301, 'date_from': '20180101'}

url = urlparse('mysql://sin:qpm4nz@localhost:3306/stockanal')

conn = mysql.connector.connect(
    host = url.hostname or 'localhost',
    port = url.port or 3306,
    user = url.username or 'root',
    password = url.password or '',
    database = url.path[1:],
)

if not conn.is_connected():
    print('connection not available.')
    exit()

sql_sellers = 'SELECT DISTINCT Short_Seller FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'
sql_sellers = sql_sellers.format(date=args['date_from'], code=args['code'])
seller_list = sorted([row[0] for row in fetch_table(conn, sql_sellers)])

sql_name = 'SELECT DISTINCT Name FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'
sql_name = sql_name.format(date=args['date_from'], code=args['code'])
name_list = sorted([row[0] for row in fetch_table(conn, sql_name)])

sql = 'SELECT Date, Short_Seller, Ratio FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'.format(date=args['date_from'], code=args['code'])
sql = sql.format(date=args['date_from'], code=args['code'])
short_table = fetch_table(conn,sql)

print(name_list,seller_list)
print(short_table[:5])

print('finished.')