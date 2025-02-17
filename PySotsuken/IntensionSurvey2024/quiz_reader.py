'''
Created on 2025/02/17

@author: sin
'''
import sys, re, pandas as pd

def read_quiz_answers(input_csv_filepath, errlog_filename='read_quiz_answer_err.log', intensions_max=24):
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

if __name__ == '__main__':
    print(sys.argv)
    if not len(sys.argv) > 2 :
        print('requires quiz anser CSV file.')
        exit(1)
    
    STUDENTS_QUIZ_FILEPATH = sys.argv[1]
    intension_df = read_quiz_answers(STUDENTS_QUIZ_FILEPATH)
    # CSVファイルに書き出す
    STUDENTS_ASSIGNMENT_INTENTION_FILEPATH = sys.argv[2]
    intension_df.to_csv(STUDENTS_ASSIGNMENT_INTENTION_FILEPATH, index=False, encoding='utf-8-sig')
    print('students\' intension has been loaded and written to '+STUDENTS_ASSIGNMENT_INTENTION_FILEPATH)
