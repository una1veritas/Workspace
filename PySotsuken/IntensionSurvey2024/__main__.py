'''
Created on 2025/01/18

@author: sin
'''
import pandas as pd
import re
import sys
from _sqlite3 import SQLITE_CONSTRAINT_ROWID

def read_quiz_answers(input_csv_filepath, intensions_max=24):
    # 括弧内の単語を抽出する正規表現パターン
    pattern = r'{(.*?)}'
    column_names = ['学生番号','氏名','メールアドレス','提出日時']
    for i in range(1, intensions_max+1) :
        column_names.append('第{0}希望'.format(i))
    df = pd.read_csv(input_csv_filepath)
    tbl = list()
    for _, row in df[['名前','身分','IDナンバ','メールアドレス','受験完了','解答 1','解答 2']].iterrows() :
        new_row = [row['IDナンバ'],row['名前'],row['メールアドレス'],row['受験完了']]
        itemstr = re.findall(pattern, row['解答 1'])
        if len(itemstr) == 0 :
            new_row.append('')
            print('error: empty 1st choice, ', new_row)
        else:
            new_row.append(itemstr[0])
        # 正規表現を使って括弧内の単語を抽出
        itemstr = re.findall(pattern, row['解答 2'])
        for t in itemstr:
            a_pair = t.split('：')
            if len(a_pair) != 2 :
                print('error: ignoring empty choice.')
                continue
            new_row.append(t)
        if len(new_row) <  len(column_names):
            for _ in range(len(new_row), len(column_names)) :
                new_row.append('')
        tbl.append(new_row)
    #print(tbl)
    return pd.DataFrame(tbl, columns=column_names)

def df_dict(df, s_column, key, v_column, err_value = None):
    row = df[df[s_column] == key][v_column]
    if row.shape[0] > 0 :
        return row.values[0]
    return err_value

# haizoku_chosa
# 入出力ファイル名
# 1. 入力：研究室情報
labs_info_filename = "labs_info-2024.csv"
# 2. 入力：学生成績等情報
STUDENTS_GRADEINFO_FILEPATH = "students_info-2024.csv"
# 3. 入力：学生の配属希望
students_preference_filename = "students_preference-2024.csv"
# 4. 入出力：教員による配属希望理由書にもとづく学生の選択
labs_preference_filename = "labs_preference.csv"
# 5. 出力：各研究室の配属情報
labs_assignments_filename = "labs_assignments.csv"
# 6. 出力：各学生の配属情報
students_assignments_filename = "students_assignments.csv"

maximum_intensions = 24
default_lab_capacity = float('inf')
default_lab_capacity_motivation = 3  # 配属希望理由書にもとづく配属人数枠

if __name__ == '__main__':
    # CSVファイルのパスを指定
    csv_file_path = 'テスト形式配属希望調査.csv'
    csv_outfile_path = 'students_intensions.csv'
    if len(sys.argv) > 1 :
        csv_file_path = sys.argv[1]
    if len(sys.argv) > 2 :
        csv_outfile_path = sys.argv[2]
    
    #intension_df = read_quiz_answers(csv_file_path)
    intension_df = pd.read_csv(students_preference_filename, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
    # CSVファイルに書き出す
    intension_df.to_csv(csv_outfile_path, index=False, encoding='utf-8-sig')
    print('students\' intension has written to '+csv_outfile_path)
        
    labs_df = pd.read_csv(labs_info_filename, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
    labs = dict() 
    # int(row['研究室番号']) : [row['研究室ラベル'],[row['指導教員名'], row['指導教員所属'], int(row['最大配属人数'])]
    for _, row in labs_df.iterrows():
        if len(row['研究室ラベル']) == 0 :
            '''ignore the row'''
            continue
        lablabel = row['研究室ラベル']
        if row['最大配属人数'] != '' and int(row['最大配属人数']) == 0 :
            continue
        elif row['最大配属人数'] != '' :  
            labs[lablabel] = [row['指導教員所属'], int(row['最大配属人数'])]
        else:
            labs[lablabel] = [row['指導教員所属'], default_lab_capacity]
    print(labs.items())

    students_df = pd.read_csv(STUDENTS_GRADEINFO_FILEPATH, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
    #print(students_df)
    students = dict()
    for _, row in students_df.iterrows() :
        sid = str(row['学生番号'])
        students[sid] = [str(row['氏名']), float(row['成績']), _] 
        intensions = intension_df.loc[intension_df['学生番号']==sid]
        if intensions.empty :
            print('error: empty intension ',sid,students[sid])
        else:
            ilist = list()
            for i in intensions.iloc[0].tolist()[4:] :
                if len(i) == 0 : continue
                a_pair = i.split('：')
                a_pair[0] = int(a_pair[0])
                ilist.append(tuple(a_pair))
            students[sid].append(ilist)
    #print(students.items())
    
    '''配属人数の決定'''
    assignments = dict()
    for key, labinfo in labs.items():
        labname = key
        assignments[labname] = dict()
        assignments[labname]['votes'] = [0 for _ in range(maximum_intensions)]
        assignments[labname]['gpa'] = [0.0 for _ in range(maximum_intensions)]
        assignments[labname]['capa'] = labinfo[1]
    for _, row in intension_df.iterrows():
        intension_list = row.tolist()[4:]
        for rank in range(len(intension_list)) :
            lab = intension_list[rank]
            if len(lab) == 0 :
                continue
            if lab not in assignments:
                print('error: not recoginized lab label', lab, rank)
                break
            popcount = assignments[lab]['votes'][rank]
            highestgpa = assignments[lab]['gpa'][rank]
            assignments[lab]['votes'][rank] = popcount+1
            assignments[lab]['gpa'][rank] = max(highestgpa, students[row['学生番号']][1])

    stats = dict()
    stats['配属希望者数数'] = intension_df.shape[0]
    stats['人数可変の研究室数'] = 0
    stats['人数固定の研究室への総配属人数'] = 0
    stats['学科外の研究室への総配属人数'] = 0
    for each in sorted(assignments.items(), reverse=True, key = lambda x: (x[1]['capa'], x[1]['votes'], x[1]['gpa'])) :
        affil = df_dict(labs_df, '研究室ラベル', each[0], '指導教員所属') 
        max_capa = df_dict(labs_df, '研究室ラベル', each[0], '最大配属人数') 
        if affil == '知能情報工学科' :
            if max_capa == '' :
                stats['人数可変の研究室数'] += 1
            else:
                stats['人数固定の研究室への総配属人数'] += int(max_capa)
        if affil != '知能情報工学科' and each[1]['votes'][0] > 0 :
            stats['学科外の研究室への総配属人数'] += 1
    stats['人数上限を+1する研究室数'] = (stats['配属希望者数数'] - stats['学科外の研究室への総配属人数'] - stats['人数固定の研究室への総配属人数']) % stats['人数可変の研究室数']
    print(stats)
    