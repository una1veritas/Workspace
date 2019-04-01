from sense_hat import SenseHat
from datetime import datetime
import time

sense = SenseHat()
sense.clear()

roomtemp = 0
roomhumidity = 0
airpressure = 0

try:
    while True:
        sense.show_message(datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                           scroll_speed=(0.1), back_colour=[0, 0, 100])
        
        temp = round(sense.get_temperature(), 1)
        humidity = round(sense.get_humidity(), 1)
        pressure = round(sense.get_pressure(), 1)
        
        if (roomtemp != temp or roomhumidity != humidity or airpressure != pressure) :
            roomtemp = temp
            roomhumidity = humidity
            airpressure = pressure
            print("Temperature C", temp)
            print("Humidity :", humidity)
            print("Pressure:", pressure)
            
            sense.show_message(str(temp) + "C, " + str(humidity) + "%, "
                               +str(pressure) + "hPa ",
                               scroll_speed=(0.1), back_colour=[0, 0, 100])

        time.sleep(5)
except KeyboardInterrupt:
    print("Exit by Ctrl-C")

sense.clear()
