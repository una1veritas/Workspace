from sense_hat import SenseHat
from datetime import datetime
import time
import statistics

def get_sensor_values(hat):
    t = hat.get_temperature()
    h = hat.get_humidity()
    p = hat.get_pressure()
    return (t, h, p)

senhat = SenseHat()
senhat.clear()

hist = dict()
hist['t'] = list()
hist['h'] = list()
hist['p'] = list()

print("Initializing sensors...", end="")
while len(hist['p']) < 4:
    vals = get_sensor_values(senhat)
    if vals[2] != 0 :
        hist['p'].append(vals[2])
        hist['t'].append(vals[0])
        hist['h'].append(vals[1])
    time.sleep(0.2)
print("done.")

last_disp = (0.0, 0.0, 0.0)
while True:
    vals = get_sensor_values(senhat)
    hist['t'].append(vals[0])
    hist['h'].append(vals[1])
    hist['p'].append(vals[2])
    hist['t'] = hist['t'][-8:]
    hist['h'] = hist['h'][-8:]
    hist['p'] = hist['p'][-8:]
    disp_values = (round(statistics.mean(hist['t']), 1), round(statistics.mean(hist['t']),2), round(statistics.mean(hist['t']),2) )
    dt_now = datetime.now()
    if (dt_now.second % 5) == 0 and disp_values != last_disp:
        print(last_measured.strftime("%Y/%m/%d %H:%M:%S"), end = ' ')
        print('{0} deg. C, {1} %, {2} hPa'.format(disp_values))
        last_disp = disp_values 
    time.sleep(0.2)
