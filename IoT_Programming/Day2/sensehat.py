from sense_hat import SenseHat
from datetime import datetime

sense = SenseHat()
sense.clear()

dt = datetime.now()

temp = round(sense.get_temperature(), 1)
humi = round(sense.get_humidity(), 1)
pres = round(sense.get_pressure(), 1)

print(dt.strftime("%Y/%m/%d %H:%M:%S"))
print('{0} deg., {1}%, {2} hPa.'.format(temp, humi, pres))

while True:
    for event in sense.stick.get_events():
        print(event.timestamp, event.direction, event.action)