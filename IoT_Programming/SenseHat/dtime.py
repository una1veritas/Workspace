from datetime import datetime, timedelta
import time
import math
import random


hist = [0,0,0]
outer = 0
width = 0.33
hists = list()
for mu in range(0, 6):
    center = mu/6
    hist = [0 for i in range(3)]
    for i in range(0,80000):
        r = random.gauss(center, width)
        if -1 <= r < 0 :
            hist[-1] += 1
        elif 0 <= r < 1 :
            hist[0] += 1
        elif 1 <= r < 2 :
            hist[1] += 1
        else :
            outer += 1
            
    hists.append([int(hist[i]/80000*280) for i in range(-1,2)])

print(hists)

exit()

idx = [22, 23, 0, 1, 2]
mag_sec = [[7, 149, 149, 7, 0], 
[4, 127, 167, 12, 0], 
[0, 103, 186, 17, 0], 
[0, 84, 197, 24, 0], 
[0, 67, 206, 34, 0], 
[0, 48, 216, 48, 0], 
[0, 34, 209, 67, 0], 
[0, 24, 198, 84, 0], 
[0, 17, 186, 103, 0], 
[0, 12, 167, 127, 4],
]

lastsec = 0.0
while True:
    dtnow = datetime.now()
    nowsec = int(4*dtnow.second + dtnow.microsecond*4//1000000) % 240
    
    if lastsec != nowsec :
        lastsec = nowsec
        print(str(lastsec) + ' ' + str(lastsec//10) + ': '+str(mag_sec[lastsec%10]))
    time.sleep(0.05)
