from sense_hat import SenseHat
from datetime import datetime

sense = SenseHat()
sense.clear()

temp = round(sense.get_temperature(), 1)
msg = str(temp)
print('Temp ' + msg + ' C deg.')
sense.show_message(msg + ' C deg.')
humi = round(sense.get_humidity(), 1)
msg = str(humi)
print('Humidity ' + msg + '% ')
sense.show_message(msg + ' %')
pres = round(sense.get_pressure(), 1)
msg = str(pres)
print('Pressure ' + msg + ' hPa ')
sense.show_message(msg + ' hPa')

dt = datetime.now()
msg = dt.strftime("%Y/%m/%d %H:%M:%S")
print(msg)
sense.show_message(msg)
