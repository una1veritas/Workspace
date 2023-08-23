import os, glob, sys

dirpath = "/Users/sin/Dropbox/検索アルゴリズム論（DS+AI+MI）/2023"

def read_registered_students(path):
# 履修登録者名簿ファイル
    table = dict()
    table['header'] = list() 
    table['cells'] = list()
    if os.path.isdir(path) :
        fdname_list = glob.glob(os.path.join(path, 'meibo*.csv'))
        if not (0 < len(fdname_list) < 2) :
            print("error: couldn't find or specify the meibo*.csv file.") 
            exit()
        meibofile = fdname_list[0]
        with open(meibofile, encoding = "utf-8-sig") as f:
            line_counter = 0
            for a_line in f:
                a_line = a_line.strip()
                if len(a_line) == 0 or a_line[0] == '#' :
                    continue # skip this an empty or a comment line.
                if line_counter == 0 :
                    # table column header
                    table['header'] = a_line.split('\t')
                else:
                    row = a_line.split(',')
                    table['cells'].append(row)
                line_counter += 1
            print(line_counter)
    return table

def inspect_reports(path):
    if not os.path.isdir(path) :
        print('error: this is not a directory path.')
        exit()
    fdname_list = glob.glob(os.path.join(path, '*'))
    assignment_folders = list()
    for each in fdname_list:
        if os.path.isdir(each):
            assignment_folders.append(os.path.basename(each))
    assignment_folders.sort()
    print(assignment_folders)

inspect_reports(dirpath)
exit()

table = read_registered_students(dirpath)
print(sorted([(each[4], each[7]) for each in table['cells']]))
print("finished.")