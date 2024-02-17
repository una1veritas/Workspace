'''
Created on 2024/01/20

@author: Sin Shimozono
'''
import sys, os, glob, re
import pypdf
from openpyxl import Workbook, load_workbook
from operator import itemgetter

#row_heading 行を heading としてシートの内容を読み出す
def btinfo_db(fname, row_heading = 1):
    try:
        wb = load_workbook(filename=fname, data_only=True)
        wsnames = wb.sheetnames
        #print(wb.sheetnames)
    except FileNotFoundError as e:
        print("{} not found.".format(fname))
        exit(1)
        return None
    ws = wb[wsnames[0]]
    heading = [ws[cellname].value for cellname in [f"{chr(ord('A')+c)}{row_heading}" for c in range(0,ws.max_column)]]
    #print(heading)
    column_dict = dict()
    for i in range(len(heading)) :
        column_dict[heading[i]] = i
    rows = []
    for row in ws.iter_rows(min_row=row_heading+1,max_row=ws.max_row,min_col=1,max_col=ws.max_column):
        values = []
        for cell in row:
            values.append(cell.value)
        rows.append(values)
        #print(values)
    #print(heading)
    return (column_dict, rows)
    
def main(argv):
    btinfofile = "";
    if len(argv) > 1 :
        btinfofile = argv[1]
        if not os.path.isfile(btinfofile) :
            print("File " + btinfofile + " does not exit.")
            exit(1)
    else:
        print(argv)
        print("No info file name.")
        exit(1)

    try:
        (btcolumn, btrows) = btinfo_db(btinfofile, 2)
        btrows.sort(key=lambda x: x[btcolumn["研究室内発表順"]]) # sort by 研究室内順番
        btrows.sort(key=lambda x: x[btcolumn["研究室発表順"]]) # sort by 発表順番
        btrows.sort(key=lambda x: x[btcolumn["発表グループ"]]) # sort by グループ
        #print(wb.sheetnames)
        # print(btcolumn)
        # for r in btrows:
        #     print(r)
        # print()
    except FileNotFoundError as e:
        print(e)
        print("Failed to read workshhet file {}.".format(btinfofile))
        exit(1)
    
    print("bye.")
    
    exit(0)


if __name__ == '__main__':
    main(sys.argv)