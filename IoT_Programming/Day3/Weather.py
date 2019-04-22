from sense_hat import SenseHat
from datetime import datetime

sense = SenseHat()
sense.clear()

temp = round(sense.get_temperature(), 1)
humi = round(sense.get_humidity(), 1)
pres = round(sense.get_pressure(), 1)

sense.show_message('Temp ' + str(temp) + 'deg ')
sense.show_message('Humidity ' + str(humi) + '% ')
sense.show_message('Pressure ' + str(pres) + ' hPa ')

dt = datetime.now()
sense.show_message(dt.strftime("%Y/%m/%d %H:%M:%S"))