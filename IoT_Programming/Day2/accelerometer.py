from sense_hat import SenseHat
import math
import time

def innerProduct(veca, vecb):
    return veca[0]*vecb[0] + veca[1]*vecb[1] + veca[2]*vecb[2]
    
sense = SenseHat()
sense.clear()

while True:
    accx = sense.get_accelerometer_raw()
    vec = [accx["x"], accx["y"], accx["z"]]
    msg = "(x: {0}, y: {1}, z: {2})".format(round(accx["x"], 3), round(accx["y"], 3), round(accx["z"], 3))
    print(msg)
    print(vec, round(innerProduct(vec, vec),2))
    if 0.98 <= innerProduct(vec, vec) <= 1.02 :
        if 0.98 <= innerProduct(vec, [0,0,1]) <= 1.02 :
            print("up")
        
    time.sleep(0.2)
