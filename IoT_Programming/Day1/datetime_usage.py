from datetime import datetime

dow = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
#現在の日時を得る
dtnow = datetime.now()

y = dtnow.year
mon = dtnow.month
d = dtnow.day
w = dtnow.weekday()
print(w)
print(y, mon, d, dow[w])

h = dtnow.hour
m = dtnow.minute
s = dtnow.second
mu = dtnow.microsecond
print(h, m, s, mu)

formatted_str = dtnow.strftime("%I:%M:%S %p, %d %b, %Y. ")
# %指定子の詳細は strftime のドキュメントを参照
print(formatted_str)