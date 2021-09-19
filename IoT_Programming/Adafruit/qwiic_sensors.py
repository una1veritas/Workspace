# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

import time
import board
from   adafruit_bme280 import basic as bme280
from   adafruit_bmp3xx import BMP3XX_I2C
from   adafruit_sht4x  import SHT4x
from   stemmaqt_veml6030 import VEML6030
#from   adafruit_tmp117 import TMP117

from   datetime import datetime, timedelta
import sys
import json

tint = 5
if len(sys.argv) > 1 :
    tint = int(sys.argv[1])
    if tint >= 60 :
        tint = (tint // 60 )*60
print("time interval: {0} sec.".format(tint))

# Create sensor object, using the board's default I2C bus.

i2c = board.I2C()  # uses board.SCL and board.SDA
bme280 = bme280.Adafruit_BME280_I2C(i2c, 0x76)
bmp380 = BMP3XX_I2C(i2c)
veml   = VEML6030(i2c, VEML6030.I2C_ADDR_ALT)
sht40  = SHT4x(i2c)
#tmp117 = TMP117(i2c)

# change this to match the location's pressure (hPa) at sea level
#bme280.sea_level_pressure = 1013.25

veml.set_gain_integtime(1,100)
print("VEML6030 current configuration: ", veml.get_gain_integtime())
time.sleep(0.5)

logfilename = 'weather.log'

logdata = list()
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
        logdata.append(dict())
        logdata[-1]["DATETIME"] = {dtnow.strftime('%Y-%m-%d'),
                                   dtnow.strftime('%H:%M:%S')}
        logdata[-1]["BME280"] = {"TC": round(bme280.temperature,2),
                                 "RH": round(bme280.relative_humidity,2),
                                 "BP": round(bme280.pressure,2)}
#        logdata[-1]["TMP117"] = {"TC": round(tmp117.temperature,2)}
        tc, rh = sht40.measurements
        logdata[-1]["SHT40"]  = {"TC": round(tc,2), "RH": round(rh,2)}
        logdata[-1]["BMP380"] = {"TC": round(bmp380.temperature,2),
                                 "BP": round(bmp380.pressure,2)}
        logdata[-1]["VEML6030"] = {"LX": round(veml.lux, 2)}
        jsontxt = json.dumps(logdata[-1])
        print(jsontxt, "\n", file = sys.stdout)
        if tint < 60 :
            dtnow = datetime(dtnow.year, dtnow.month, dtnow.day,
                             dtnow.hour, dtnow.minute, int((dtnow.second//tint)*tint) )
        else:
            tmin = tint // 60
            dtnow = datetime(dtnow.year, dtnow.month, dtnow.day,
                             dtnow.hour, int((dtnow.minute//tmin)*tmin), 0)
        dtnext = dtnow + dtdelta
        time.sleep(2)
    if len(logdata) :
        try:
            with open(logfilename, mode='a') as lfile:
                jsontxt = json.dumps(logdata[-1])
                lfile.write(jsontxt)
                lfile.write("\n")
                logdata.pop(0)
        except Exception as x:
            print(x, file=sys.stderr)
        
    time.sleep(0.4)
