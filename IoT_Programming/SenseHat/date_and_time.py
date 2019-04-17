from sense_hat import SenseHat
from datetime import datetime, timezone, timedelta

s_hat = SenseHat()
s_hat.clear()

tzone_JST = timezone(timedelta(hours=+9), 'JST')
tzone_UCT = timezone(timedelta(hours=0), 'UCT')

first_time = True
watch_last = datetime.now(tzone_JST)

try:
    while True:
        temp = datetime.now(tzone_JST)
        msg = ""
        if first_time or (temp.hour != watch_last.hour) or \
           (temp.minute != watch_last.minute and temp.minute % 15 == 0) :
            watch_last = temp
            msg = watch_last.strftime("%H:%M:%S ") \
                  + watch_last.strftime("%A")[:3] \
                  + watch_last.strftime(" %Y/%m/%d")
            s_hat.show_message(msg, 0.1, [127,127,127], [40,40,40])
            first_time = False
            watch_last = datetime.now(tzone_JST)
        if temp.minute != watch_last.minute or \
           (temp.second != watch_last.second) and temp.second % 15 == 0 :
            watch_last = temp
            msg = watch_last.strftime("%H:%M:%S")
            s_hat.show_message(msg, 0.1, [127,127,127], [40,40,40])
            watch_last = datetime.now(tzone_JST)
#        elif temp.second != watch_last.second :
#            watch_last = temp
#            msg = watch_last.strftime("%S")
#            s_hat.show_message(msg, 0.04, [127,127,127], [40,40,40])
#            watch_last = datetime.now(tzone_JST)

except KeyboardInterrupt:
    pass

s_hat.clear()
