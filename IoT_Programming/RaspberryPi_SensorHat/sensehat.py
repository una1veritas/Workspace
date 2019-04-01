from sense_hat import SenseHat
import time

sense = SenseHat()
sense.clear()

while True :
    temp = sense.get_temperature()
    humi = sense.get_humidity()
    pres = sense.get_pressure()
    
    msg = u"Temp. {0} °C. Humidity {1} % Air pressure {2} hPa".format(round(temp,1), round(humi,1), round(pres,1))
    print(msg)
    
    accx = sense.get_accelerometer_raw()
    msg = "(x: {0}, y: {1}, z: {2})".format(round(accx["x"], 3), round(accx["y"], 3), round(accx["z"], 3))
    print(msg)

    for event in sense.stick.get_events():
        print(event.timestamp, event.direction, event.action)
    
    time.sleep(3)
    
