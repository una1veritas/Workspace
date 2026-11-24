'''
Created on 2026/11/16

@author: sin
'''
import csv
import re
import sys

fname = sys.argv[1]
with open(fname, "r", encoding="utf-8", errors="ignore") as f:
  text = f.read().strip()

# 1つ以上の空行（改行2つ以上）でレコード分割
records = re.split(r"\n\s*\n", text)

rows = []
for rec in records:
  # 各レコード内の改行でフィールド分割（前後の余分な空白は除去）
  fields = [line.strip() for line in rec.splitlines() if line.strip()]
  if fields:
    rows.append(fields)

# Excelでも文字化けしないように utf-8-sig で出力
with open("output.csv", "w", encoding="utf-8", newline="") as f:
  writer = csv.writer(f)
  writer.writerows(rows)

print(f"{len(rows)} 件のレコードを出力しました。")

if __name__ == '__main__':
    pass
