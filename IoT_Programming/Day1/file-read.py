import sys

filename = "data.csv"
table = []
with open(filename, 'r') as file:
    for line in file:
        line = line.strip()
        items = line.split(',')
        items[1] = float(items[1])
        items[2] = float(items[2])
        items[3] = float(items[3])
        table.append(items[:4])   # 最初の 4 要素のみを行として追加
# with open のブロックから出ると，ファイルは閉じられる．

print(table[:10])     # 最初のデータ 10 行分を印字