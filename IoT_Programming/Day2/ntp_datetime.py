# -*- coding: utf-8 -*-
from datetime import datetime, timedelta, timezone

dt = datetime.now()
print(dt.strftime("%Y/%m/%d %H:%M:%S"))
print('micro sec. ' + str(dt.time().microsecond) )

# 日本標準時，世界標準時など時差を使う
TZ_JST = timezone(timedelta(hours=+9), 'JST') #　日本（明石）標準時
TZ_UTC = timezone(timedelta(hours=0), 'UTC') #　世界（グリニジ）標準時
dt = datetime.now(TZ_JST)
print(dt.tzname())
print(dt.strftime("%Y/%m/%d %H:%M:%S"))
print('micro sec. ' + str(dt.time().microsecond) )

print("UTC " + str(dt.astimezone(TZ_UTC)))