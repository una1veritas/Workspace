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
from pickle import NONE

def trunc_time(dtnow, delta):
    dmin = delta.seconds // 60
    dsec = delta.seconds % 60
    if dmin > 0 :
        return datetime(dtnow.year, dtnow.month, dtnow.day, dtnow.hour,
                        int(dtnow.minute//dmin)*dmin, 0)
    elif dsec > 0 :
        return datetime(dtnow.year, dtnow.month, dtnow.day, dtnow.hour,
                        dtnow.minute, int(dtnow.second//dsec)*dsec)
    else:
        return dtnow
    

tmin = 5
if len(sys.argv) > 1 :
    tmin = max(tmin, int(sys.argv[1]))
print("record time interval: {0} min.".format(tmin))
rintervals = timedelta(minutes=tmin)
# Create sensor object, using the board's default I2C bus.

i2c = board.I2C()  # uses board.SCL and board.SDA
bme280 = bme280.Adafruit_BME280_I2C(i2c, 0x76)
bmp380 = BMP3XX_I2C(i2c)

try:
    veml = VEML6030(i2c, VEML6030.I2C_ADDR_ALT)
    veml.set_gain_integtime(1,200)
    time.sleep(0.5)
except ValueError as exc:
    veml = None
sht40  = SHT4x(i2c)

#tmp117 = TMP117(i2c)

# change this to match the location's pressure (hPa) at sea level
#bme280.sea_level_pressure = 1013.25


logfilename = 'weatherlog.json'
mintervals = timedelta(seconds=15)

logdata = list()
dtnow = datetime.now()
dtrecorded = trunc_time(dtnow, rintervals)
dtmeasured = trunc_time(dtnow, mintervals)


while True:
    dtnow = datetime.now()
    if dtnow >= (dtmeasured + mintervals):
        logdata.append(dict())
        logdata[-1]["TIMESTAMP"] = dtnow.strftime('%Y-%m-%d %H:%M:%S')
        logdata[-1]["BME280.TC"] = bme280.temperature
        logdata[-1]["BME280.RH"] = bme280.relative_humidity
        logdata[-1]["BME280.BP"] = bme280.pressure
        #logdata[-1]["TMP117"] = {"TC": round(tmp117.temperature,2)}
        tc, rh = sht40.measurements
        logdata[-1]["SHT40.TC"]  = tc
        logdata[-1]["SHT40.RH"] = rh
        logdata[-1]["BMP380.TC"] = bmp380.temperature
        logdata[-1]["BMP380.BP"] = bmp380.pressure
        if veml: logdata[-1]["VEML6030.LX"] = veml.lux 
        jsontxt = json.dumps(logdata[-1])
        print(jsontxt, "\n", file = sys.stdout)
        #
        print(dtrecorded.strftime('%Y-%m-%d %H:%M:%S'), dtmeasured.strftime('%Y-%m-%d %H:%M:%S'), dtnow.strftime('%Y-%m-%d %H:%M:%S'))
        dtmeasured = trunc_time(dtnow, mintervals)

    if dtnow >= dtrecorded + rintervals and len(logdata) :
        try:
            with open(logfilename, mode='a') as lfile:
                avrgdata = logdata[-1]
                count = 1
                dtback = dtrecorded + rintervals - timedelta(minutes=5)
                for data in logdata[:-1]:
                    if data["TIMESTAMP"] < dtback.strftime('%Y-%m-%d %H:%M:%S') : continue
                    for key in data:
                        if key == "TIMESTAMP" : continue
                        avrgdata[key] += data[key]
                    count += 1
                print("count = ", count)
                for key in avrgdata:
                    if key == "TIMESTAMP" : continue
                    avrgdata[key] = round(avrgdata[key]/count, 2)
                dtrecorded = trunc_time(dtnow, rintervals)
                avrgdata["TIMESTAMP"] = dtrecorded.strftime('%Y-%m-%d %H:%M:%S')
                jsontxt = json.dumps(avrgdata)
                lfile.write(jsontxt)
                lfile.write("\n")
                print(jsontxt)
                logdata.clear()
        except Exception as x:
            print(x, file=sys.stderr)

    time.sleep(5)
