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

sql = 'SELECT DISTINCT Short_Seller FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'.format(date='2018-09-01', code=5301)
seller_table = sorted([row[0] for row in fetch_table(conn,sql) ])
print(len(seller_table), seller_table[:10])

sql = 'SELECT Date, Short_Seller, Ratio FROM short_positions WHERE Date >= \'{date}\' AND Code = {code}'.format(date='2018-09-01', code=5301)
short_table = fetch_table(conn,sql)
print(short_table[:5])
print('finished.')