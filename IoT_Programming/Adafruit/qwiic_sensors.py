# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
from adafruit_bme280 import basic as adafruit_bme280
import adafruit_bmp3xx
import adafruit_sht4x
from stemmaqt_veml6030 import import VEML6030
import adafruit_tmp117

from datetime import datetime,timedelta
import sys

tint = 5
if len(sys.argv) > 1 :
    tint = int(sys.argv[1])
    if tint >= 60 :
        tint = (tint // 60 )*60
print("time interval: {0} sec.".format(tint))

# Create sensor object, using the board's default I2C bus.

i2c = board.I2C()  # uses board.SCL and board.SDA
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c, 0x76)
bmp380 = adafruit_bmp3xx.BMP3XX_I2C(i2c)
veml =  VEML6030(i2c, VEML6030.I2C_ADDR_ALT)
sht40 = adafruit_sht4x.SHT4x(i2c)
tmp117 = adafruit_tmp117.TMP117(i2c)

# change this to match the location's pressure (hPa) at sea level
bme280.sea_level_pressure = 1013.25

print("configuration: ", hex(veml.configuration))
veml.set_gain_integtime(1,200)
print("new configuration: ", hex(veml.configuration))
time.sleep(0.5)

dtdelta = timedelta(seconds=tint)
dtnow = datetime.now()
if tint < 60 :
    dtnow = datetime(dtnow.year, dtnow.month, dtnow.day,
                     dtnow.hour, dtnow.minute, int((dtnow.second//tint)*tint) )
else:
    tmin = tint // 60
    dtnow = datetime(dtnow.year, dtnow.month, dtnow.day,
                     dtnow.hour, int((dtnow.minute//tmin)*tmin), 0)
dtnext = dtnow + dtdelta

while True:
    dtnow = datetime.now()
    if dtnow >= dtnext:
        print(dtnow.strftime('%Y-%m-%d %H:%M:%S'))
        print("BME280: %0.1f C, %0.1f %%, %0.1f hPa"
              % (bme280.temperature, bme280.relative_humidity, bme280.pressure))
        print("TMP117: %0.1f C" % tmp117.temperature)
        print("SHT40: %0.1f C, %0.1f %%" % sht40.measurements)
        print("BMP380: %0.1f hPa, %0.1f C" % (bmp380.pressure, bmp380.temperature))
#        print("BMP380: %0.1f hPa, %0.1 deg.C"
#              % (bmp380.pressure, bmp380.temperature))
        print("VEML6030: %0.1f lux" % veml.lux)
        #    print("Altitude = %0.2f meters" % bme280.altitude)
        print()
        if tint < 60 :
            dtnow = datetime(dtnow.year, dtnow.month, dtnow.day,
                             dtnow.hour, dtnow.minute, int((dtnow.second//tint)*tint) )
        else:
            tmin = tint // 60
            dtnow = datetime(dtnow.year, dtnow.month, dtnow.day,
                             dtnow.hour, int((dtnow.minute//tmin)*tmin), 0)
        dtnext = dtnow + dtdelta
        time.sleep(3)
    time.sleep(0.5)
