# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
from adafruit_bme280 import basic as adafruit_bme280
import adafruit_bmp3xx
import adafruit_sht4x
import vishay_veml6030
import adafruit_tmp117

from datetime import datetime
# Create sensor object, using the board's default I2C bus.

i2c = board.I2C()  # uses board.SCL and board.SDA
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c, 0x76)
bmp380 = adafruit_bmp3xx.BMP3XX_I2C(i2c)
veml =  vishay_veml6030.VEML6030(i2c, vishay_veml6030.VEML6030.I2C_ADDR_ALT)
sht40 = adafruit_sht4x.SHT4x(i2c)
tmp117 = adafruit_tmp117.TMP117(i2c)

# change this to match the location's pressure (hPa) at sea level
bme280.sea_level_pressure = 1013.25

print("configuration: ", hex(veml.configuration))
veml.gain = 1/2
veml.integration_time = 100
print("new configuration: ", hex(veml.configuration))
time.sleep(0.5)


while True:
    dtnow = datetime.now()
    if dtnow.second % 30 == 0 :
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
    time.sleep(1)
