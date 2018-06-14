
path = 'graph.txt'

try:
    # ファイルオブジェクトを変数fに代入。close()は書かなくてＯＫ。
    with open(path, mode='r', encoding='utf-8') as t_file:
        for a_line in t_file:
            print(a_line.split())
except:
    print(path + 'の読み込みに失敗しました。')
