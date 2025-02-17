'''
Created on 2025/01/18

@author: sin
'''
import pandas as pd
import re, sys

# haizoku_chosa
# 入出力ファイル名
# 1. 入力：研究室情報
LABS_INFO_FILEPATH = "labs_info-2024.csv"
# 2. 入力：学生成績等情報
STUDENTS_GRADEINFO_FILEPATH = "GPA_b3_20240926.csv"
# 3. 入力：学生の配属希望
STUDENTS_ASSIGNMENT_INTENTION_FILEPATH = "students_intentions-2024.csv"
# 4. 入出力：教員による配属希望理由書にもとづく学生の選択
SUPERVISORS_INTENTION_FILEPATH = "labs_preference.csv"
# 5. 出力：各研究室の配属情報
labs_assignments_filename = "labs_assignments.csv"
# 6. 出力：各学生の配属情報
students_assignments_filename = "students_assignments.csv"

max_intentionlist_length = 24
default_lab_capacity = float('inf')
default_lab_capacity_bymotivation = 3  # 配属希望理由書にもとづく配属人数枠

def df_dict(df, s_column, key, v_column, err_value = None):
    row = df[df[s_column] == key][v_column]
    if row.shape[0] > 0 :
        return row.values[0]
    return err_value

# Define a function to convert values to int or float
def try_to_numeric(value):
    try:
        return int(value)
    except ValueError:
        if len(str(value)) == 0 :
            return float('inf')
        try:
            return float(value)
        except ValueError:
            return value

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
            if t == '20：選択しない' or len(t.strip()) == 0 :
                print('error: ignoring empty choice.')
                continue
            new_row.append(t)
        if len(new_row) <  len(column_names):
            for _ in range(len(new_row), len(column_names)) :
                new_row.append('')
        tbl.append(new_row)
    #print(tbl)
    return pd.DataFrame(tbl, columns=column_names)

def df_to_dict(df, index_column, value_columns = []):
    dy = dict()
    for _, row in df.iterrows():
        if len(row[index_column]) == 0 :
            '''ignore the row'''
            continue
        key = row[index_column]
        dy[key] = [row[col] for col in value_columns]
    return dy

if __name__ == '__main__':
    args = sys.argv[1:]
    
    intension_df = pd.read_csv(STUDENTS_ASSIGNMENT_INTENTION_FILEPATH, encoding='utf-8-sig', keep_default_na=False, na_filter=False)
    print('students\' intension has been loaded from ' + STUDENTS_ASSIGNMENT_INTENTION_FILEPATH)
    
    # 研究室番号, 研究室ラベル, 指導教員名, 指導教員所属, 最大配属人数, 知能情報工学概論での研究室紹介
    labs_df = pd.read_csv(LABS_INFO_FILEPATH, converters={'最大配属人数':try_to_numeric}, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
    labs_df = labs_df.drop(labs_df.index[labs_df['最大配属人数'] == 0].tolist())
    labs_df = labs_df.drop(labs_df.index[labs_df['研究室ラベル'] == ''].tolist())
    #print(labs_df)
    labs = df_to_dict(labs_df, '研究室ラベル', ['指導教員所属', '最大配属人数'])
    print(labs_df)

    students_df = pd.read_csv(STUDENTS_GRADEINFO_FILEPATH, encoding='utf-8-sig',keep_default_na=False, na_filter=False)
    
    print('columns = ', students_df.columns)
    #students_df = students_df[['学生番号', '学生氏名', '学年', '通計GPA評価']]
    students_df.rename(columns={'通計GPA評価': 'GPA', 'GPA値': 'GPA', 'ＧＰＡ値': 'GPA', '学生氏名': '氏名'}, inplace=True)
    print('students_df columns = ', students_df.columns)
    students_df = students_df[['学生番号', '氏名', 'GPA']]
    print(students_df.columns)
    students_df.to_csv('students_grade_info-2024.out.csv', index=False)
    #print(students_df)
    #exit(1)
    
    '''配属人数の決定'''
    assignments = dict()
    for key, info in labs.items():
        labname = key
        assignments[labname] = dict()
        assignments[labname]['votes'] = [0 for _ in range(max_intentionlist_length)]
        assignments[labname]['gpa'] = [0.0 for _ in range(max_intentionlist_length)]
        assignments[labname]['capa'] = info[1]
    for _, row in intension_df.iterrows():
        intension_list = row.tolist()[4:]
        for rank in range(len(intension_list)) :
            lab = intension_list[rank]
            if len(lab) == 0 :
                continue
            if lab not in assignments:
                print('error: not recoginized lab label {} in rank {}.'.format(lab, rank) )
                break
            popcount = assignments[lab]['votes'][rank]
            highestgpa = assignments[lab]['gpa'][rank]
            assignments[lab]['votes'][rank] = popcount+1
            if df_dict(students_df,'学生番号', row['学生番号'], 'GPA') == None :
                print(f'エラー: GPA が見つからない {row["学生番号"]} {row["氏名"]}')
                print(students_df[students_df['学生番号'] == row['学生番号']])
            else:
                assignments[lab]['gpa'][rank] = float(max(highestgpa, df_dict(students_df,'学生番号', row['学生番号'], 'GPA')))
    
    with open('vote-stats.csv', mode='w', encoding='utf-8-sig') as f :
        f.write('研究室ラベル,') 
        for i in range(1,max_intentionlist_length):
            f.write(f'第{i}希望,')
        f.write('総希望者数'+ '\n')
        for key, value in assignments.items() :
            f.write('{0}'.format(key))
            for i in range(max_intentionlist_length):
                if i < len(value['votes']) :
                    f.write(',{0}'.format(value['votes'][i]))
                else:
                    f.write(',0')
            tally = sum(value['votes'])
            f.write(',{0}'.format(tally))
            f.write('\n')
    
    stats = dict()
    stats['配属希望者数数'] = intension_df.shape[0]
    stats['人数可変の研究室数'] = 0
    stats['人数固定の研究室への総配属人数'] = 0
    stats['学科外の研究室への総配属人数'] = 0
    for each in sorted(assignments.items(), reverse=True, key = lambda x: (x[1]['capa'], x[1]['votes'], x[1]['gpa'])) :
        affil = df_dict(labs_df, '研究室ラベル', each[0], '指導教員所属') 
        max_capa = df_dict(labs_df, '研究室ラベル', each[0], '最大配属人数') 
        if affil == '知能情報工学科' :
            if max_capa == float('inf') :
                stats['人数可変の研究室数'] += 1
            else:
                stats['人数固定の研究室への総配属人数'] += int(max_capa)
        if affil != '知能情報工学科' and each[1]['votes'][0] > 0 :
            stats['学科外の研究室への総配属人数'] += 1
    stats['基本配属人数'] = (stats['配属希望者数数'] - stats['学科外の研究室への総配属人数'] - stats['人数固定の研究室への総配属人数']) // stats['人数可変の研究室数']
    stats['人数上限を+1する研究室数'] = (stats['配属希望者数数'] - stats['学科外の研究室への総配属人数'] - stats['人数固定の研究室への総配属人数']) % stats['人数可変の研究室数']
    print(stats)
    '''研究室をソート'''
    for key, val in sorted(assignments.items(), reverse=True, key=lambda item: (item[1]['votes'], item[1]['gpa']) ) :
        print(key,val )
