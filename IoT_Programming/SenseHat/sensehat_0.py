from sense_hat import SenseHat
from datetime import datetime

sense = SenseHat()
sense.clear()

temp = sense.get_temperature()
humi = sense.get_humidity()
pres = sense.get_pressure()
    
print(datetime.now())
print(temp, humi, pres)
