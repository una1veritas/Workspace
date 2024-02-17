'''
Created on 2024/01/20

@author: Sin Shimozono
'''
import sys, os, glob, re

def db_column(db, col):
    column = list()
    heading = dict()
    if type(col) is str:
        for i in range(len(db[0])) :
            heading[db[0][i]] = i
        col = heading[col]

    column.append(db[0][col])
    for row in db[1:] :
        column.append(row[col])
    return column

def db_sids(db):
    sidset = set()
    sidlist = list()
    repatt = re.compile('^[0123456789C-]+$')
    for colname in db[0] :
        res = re.match(repatt, colname)
        if res is not None :
            sidstr = colname.split('-')[0]
            if sidstr not in sidset :
                sidlist.append(sidstr)
                sidset.add(sidstr)
    return sidlist

def db_from_file(filepath):
    db = list()
    repatt = re.compile('\([0123456789C-]+\)')
    with open(filepath, mode="r", encoding='utf-8') as csvf :
        for a_line in csvf :
            a_line = a_line.strip()
            if len(db) == 0:
                db.append(list())
                for colname in a_line.split(','):
                    res = repatt.search(colname)
                    if res is not None :
                        colname = res.group()[1:-1]
                    db[-1].append(colname)
            else:
                a_row = a_line.split(',')
                db.append(a_row)
    return db

def main(argv):
    csvbasedir = u"."
    if len(argv) > 1 :
        csvbasedir = argv[1]
    else:
        print(argv)
        print("No directry name.")
        exit(1)
    if not os.path.isdir(csvbasedir) :
        print("Directory" + csvbasedir + " does not exit.")
        exit(1)
    
    searchpath = os.path.join(csvbasedir, u"**", u"*.csv")
    paths = glob.glob(searchpath, recursive=True)
    if len(paths) == 0 :
        print("no csv files found.")
        exit(0)

    dbs = dict()
    for filepath in paths :
        filename = os.path.basename(filepath)
        db = db_from_file(filepath)
        dbs[filename] = db
    
    for (key, db) in dbs.items():
        print(key)
        for a_row in db:
            print(a_row)
    sids = db_sids(dbs['hyoka1.csv'])
    for sid in sids :
        print(sid)
        for colname in dbs['hyoka1.csv'][0] :
            if colname.startswith(sid) :
                for each in db_column(dbs['hyoka1.csv'], colname):
                    print(each)

if __name__ == '__main__':
    main(sys.argv)