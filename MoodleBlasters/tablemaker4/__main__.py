import os, glob, sys

if os.name == 'posix' :
    dirpath = u"/Users/sin/Dropbox/オートマトンと言語理論/2024/演習課題/"
elif os.name == 'nt' :
    dirpath = u'C:\\Users\\Sin Shimozono\\Dropbox\\検索アルゴリズム論（DS+AI+MI）\\2023'

def read_registered_students(path, meibofile = 'registered*.csv'):
# 履修登録者名簿ファイル
    table = dict()
    table['header'] = list() 
    table['cells'] = list()
    if os.path.isdir(path) :
        fdname_list = glob.glob(os.path.join(path, meibofile))
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
                    table['header'] = a_line.split(',')
                else:
                    row = a_line.split(',')
                    table['cells'].append(row)
                line_counter += 1
            #print(line_counter)
    #print(table)
    return table

def inspect_reports(path):
    if not os.path.isdir(path) :
        print('error: this is not a directory path.')
        exit()
    table = dict()
    fdname_list = glob.glob(os.path.join(path, '*'))
    assignment_folders = list()
    for each in fdname_list:
        if os.path.isdir(each):
            assignment_folders.append(os.path.basename(each))
    assignment_folders.sort()
    for folder in assignment_folders :
        folderpath = os.path.join(path, folder)
        table[folder] = dict()
        for filepath in glob.glob(os.path.join(folderpath, '*')) :
            filename = os.path.basename(filepath)
            sid = filename.split('_')[0]
            #print(folder, sid, os.path.getsize(filepath))
            table[folder][sid] = os.path.getsize(filepath)
    return table

def make_table(registered, reports):
    students = list()
    assignments = list()
    sidpos = -1
    #print(registered['header'])
    for pos in range(len(registered['header'])):
        if registered['header'][pos] == u'学生番号' or \
        registered['header'][pos] == 'sid' :
            sidpos = pos
            break
    if sidpos == -1 :
        print('error: can\'t find id in regtable header.')
        exit()
    for entry in registered['cells']:
        students.append(entry[sidpos])
    students.sort()
    #print(students)
    assignments = sorted(reports.keys())
    #print(assignments)
    
    result = list()
    ref = dict()
    result.append([u'学生番号'])
    # タイトル行　課題名
    for each in assignments :
        result[-1].append(each)
    for each in students:
        result.append([each])
        ref[each] = result[-1]
        for i in range(len(assignments)) :
            result[-1].append(0)
    #print(ref)
    for assign_id in range(len(reports)):
        for sid in students:
            if sid in reports[assignments[assign_id]] :
                assignname = assignments[assign_id]
                submsize = reports[assignname][sid]
                ref[sid][1+assign_id] = submsize
    return result

print(sys.argv)
if (len(sys.argv) > 1) :
    dirpath = sys.argv[1]
if len(sys.argv) > 2 :
    meibofile = sys.argv[2]
outfile = 'result.txt'
if len(sys.argv) > 3 :
    outfile = sys.argv[3]
regtable = read_registered_students(dirpath, meibofile)
print(regtable)
repotable = inspect_reports(dirpath)
print(repotable)
table = make_table(regtable, repotable)
with open(outfile, mode = 'w') as outfile:
    for row in table:
        outfile.write(row[0])
        if len(row) > 1 :
            for i in range(1, len(row)):
                outfile.write('\t')
                outfile.write(str(row[i]))
        outfile.write('\n')
        
print("finished.")