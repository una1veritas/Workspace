#coding: utf-8

# https://www.raspberrypi.org/documentation/hardware/sense-hat/
# go to "Calibration"
# do after the line "cp /usr/share/librtimulib-utils/RTEllipsoidFit ./ -a"
# https://qiita.com/mt08/items/947d09654394c85ad0de

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

