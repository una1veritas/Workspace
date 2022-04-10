h = ('まさし', (9, 6), 'みなこ', (4, 3))
for i in h:
    print(i)
(y, t) = h[1]
print(str(y)+'年ぶり'+ str(t)+'回目')

l = list(h)   # 組だと変更できないのでリストに変換
l.append('のりこ')
print(l)  #l 自身が変更されていることを確認
l[4] = 'のりゆき'  #リストは添え字の位置に代入可能．
print(l)
last = l.pop()  # リストの最後の要素を取り出し削除
print(last)
print(l)
l2 = l + ['のりこ', (0, 1)] # + で連結演算ができる. できる列は新しくつくられた列
print(l)
print(l2)
