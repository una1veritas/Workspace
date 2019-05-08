from sense_hat import SenseHat
import time

hat = SenseHat()
hat.clear()

lasttime = time.time()

# set the initial value
ori = hat.get_orientation()
pitch = round(ori['pitch'],3)
roll = round(ori['roll'],3)
yaw = round(ori['yaw'],3)

while True:
    ori = hat.get_orientation()
    pitch = (pitch + ori['pitch'])/2
    roll = (pitch + ori['roll'])/2
    yaw = (yaw + ori['yaw'])/2

    if time.time() - lasttime > 0.5 :
        lasttime = time.time()
        print(' pitch {0}  roll {1}  yaw {2}'.format(
            round(pitch,4),
            round(roll,4),
            round(yaw, 4)))
