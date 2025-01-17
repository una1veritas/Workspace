'''
Created on 2025/01/18

@author: sin
'''
import pandas as pd
import re
import sys

if __name__ == '__main__':
    # CSVファイルのパスを指定
    csv_file_path = 'テスト形式配属希望調査.csv'
    if len(sys.argv) > 1 :
        csv_file_path = sys.argv[1]
    out_csv_filename = '.'.join(csv_file_path.split('.')[:-1]) + '.out.csv'
    print(out_csv_filename)
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file_path)
    # 括弧内の単語を抽出する正規表現パターン
    table = list()
    pattern = r'{(.*?)}'
    for row_index, row in df[['IDナンバ','名前','解答 1','解答 2']].iterrows() :
        intenseq = list()
        intenseq.append(row['IDナンバ'])
        words = re.findall(pattern, row['解答 1'])
        if len(words) == 0 :
            intenseq.append('N.A.')
            print('error')
        else:
            intenseq.append(words[0])
        # 正規表現を使って括弧内の単語を抽出
        words = re.findall(pattern, row['解答 2'])
        # 抽出した単語を表示
        for t in words:
            thepair = t.split('：')
            if len(thepair) != 2 :
                continue
            intenseq.append(t)
        if len(intenseq) > 0 :
            table.append(intenseq)
    with open(out_csv_filename, 'w', encoding='utf-8-sig') as outf :
        columnnames = ['学生番号'] + ['第{0}希望'.format(i) for i in range(1,21)]
        outf.write(','.join(columnnames))
        outf.write('\n')
        for r in table:
            outf.write(','.join(r))
            outf.write('\n')
    print('finished.')
    