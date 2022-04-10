r = range(-2,3)
s = set(r)   # 範囲型 range を集合型に変換
print(s)
u = {0, 2, 4, 6, -2, -4}
t = s & u
t.add(4)
print(t)

d = {"apple": "リンゴ", "this": "これ", "is a": "は", "is the": "が", "dog" : "いぬ"}
print(d)
print(d.keys())
d["pen"] = "ペン"
msg = "this is a pen"
for key in d: 
   if key in msg :
      print(key, d[key])
