from sense_hat import SenseHat
import time

sense = SenseHat()
sense.clear()

while True:
    temp = sense.get_temperature()
    humi = sense.get_humidity()
    pres = sense.get_pressure()
    
    print("temp., humid., air press. = ", round(temp,1), round(humi,1), round(pres,1))
    
    acc = sense.get_accelerometer_raw()
    ori = sense.get_orientation()
    
    print(ori)
    print("Acceleration = ({0}, {1}, {2})".format(round(acc["x"], 3), round(acc["y"], 3), round(acc["z"], 3)) )
    print("Orientation = ({0}, {1}, {2})".format(round(ori["yaw"], 3), round(ori["pitch"], 3), round(ori["roll"], 3)) )

    for event in sense.stick.get_events():
        print(event.direction, event.action)
    
    time.sleep(5)
    
