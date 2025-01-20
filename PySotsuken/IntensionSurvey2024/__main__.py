'''
Created on 2025/01/18

@author: sin
'''
import pandas as pd
import re
import sys

def read_quiz_answers(csv_file_path, columns=25):
    df = pd.read_csv(csv_file_path)
    # 括弧内の単語を抽出する正規表現パターン
    pattern = r'{(.*?)}'
    tbl = list() # as list of lists
    for _, row in df[['名前','身分','IDナンバ','メールアドレス','受験完了','解答 1','解答 2']].iterrows() :
        newrow = [row['名前'],row['身分'],row['IDナンバ'],row['メールアドレス'],row['受験完了']]
        words = re.findall(pattern, row['解答 1'])
        if len(words) == 0 :
            newrow.append('N.A.')
            print('error')
        else:
            newrow.append(words[0])
        # 正規表現を使って括弧内の単語を抽出
        words = re.findall(pattern, row['解答 2'])
        for t in words:
            thepair = t.split('：')
            if len(thepair) != 2 :
                continue
            newrow.append(t)
        if len(newrow) < columns :
            for _ in range(len(newrow), columns) :
                newrow.append('')
        tbl.append(newrow)
    return tbl


if __name__ == '__main__':
    # CSVファイルのパスを指定
    csv_file_path = 'テスト形式配属希望調査.csv'
    if len(sys.argv) > 1 :
        csv_file_path = sys.argv[1]
    out_csv_filename = '.'.join(csv_file_path.split('.')[:-1]) + '.out.csv'
    print(out_csv_filename)
    tbl = read_quiz_answers(csv_file_path, 25)
    colnames = ['氏名','グループ','学生番号','メールアドレス','日付']
    for i in range(1, 25-len(colnames)+1) :
        colnames.append('第{0}希望'.format(i))
    df = pd.DataFrame(tbl, columns=colnames)
    # CSVファイルに書き出す
    # df.to_csv(out_csv_filename, index=True, encoding='utf-8-sig')
    with open(out_csv_filename, mode='w', encoding='utf-8-sig') as f :
        f.write(','.join(df.columns.tolist()))
        f.write('\n')
        for _, row in df.iterrows():
            f.write(','.join(row.tolist()))
            f.write('\n')                
    
    print('finished.')