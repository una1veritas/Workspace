'''
Created on 2020/01/30

@author: sin
'''
text= 'はと 名詞 動物\nまめ 名詞 植物\nます 名詞 動物'
dataset = list()
for w in text.split('\n'):
    dataset.append(w.split())
print(dataset)
s = set()
for t in dataset:
    s.add(tuple(t))
print(s)