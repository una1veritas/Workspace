# -*- coding: utf-8 -*-
from datetime import datetime, timedelta, timezone

TZ_JST = timezone(timedelta(hours=+9), 'JST')
TZ_UTC = timezone(timedelta(hours=0), 'UTC')
dt_now = datetime.now(TZ_JST)
print(dt_now.tzname())
print(dt_now.strftime("%Y/%m/%d %H:%M:%S"))
print('micro sec. ' + str(dt_now.time().microsecond) )

print(dt_now.astimezone(TZ_UTC))