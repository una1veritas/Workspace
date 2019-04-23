from sense_hat import SenseHat
from datetime import datetime

sense = SenseHat()
sense.clear()

temp = round(sense.get_temperature(), 1)
msg = 'Temp ' + str(temp) + 'deg '
print(msg)
sense.show_message(msg)
humi = round(sense.get_humidity(), 1)
msg = 'Humidity ' + str(humi) + '% '
print(msg)
sense.show_message(msg)
pres = round(sense.get_pressure(), 1)
msg = 'Pressure ' + str(pres) + ' hPa '
print(msg)
sense.show_message(msg)

dt = datetime.now()
msg = dt.strftime("%Y/%m/%d %H:%M:%S")
print(msg)
sense.show_message(msg)
