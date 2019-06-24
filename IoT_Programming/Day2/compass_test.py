from sense_hat import SenseHat
import time

hat = SenseHat()
hat.clear()

hatdir = 0.0

while True :
    tmpdir = round(hat.get_compass(),1)
    if ( hatdir != tmpdir ) :
        hatdir = tmpdir
        print('{:03.1f}'.format(hatdir))
    time.sleep(0.1)

