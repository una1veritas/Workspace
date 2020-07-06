from sense_hat import SenseHat
import time
from datetime import datetime

# Conway's Game of Life
# J. H. Conway, 1937 -- 2020

sense = SenseHat()
sense.clear()

world = [int(s) for s in \
    '00111000' \
    '00111000' \
    '00010000' \
    '00000000' \
    '00000000' \
    '00000000' \
    '00000000' \
    '00000000']

colors = [[0,0,0], [130,130,130]]

stamp = datetime.now().timestamp()
while True:
    scrn = [colors[0] if c == 0 else colors[1] for c in world]
    sense.set_pixels(scrn)
    time.sleep(1)
    newworld = world.copy()
    for i in range(0,64):
        livecount = 0
        for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
            if 0 <= i+d and i+d < 64 :
                if world[i+d] != 0 :
                    livecount += 1
        if world[i] == 0 and livecount == 3 :
            newworld[i] = 1
        elif world[i] == 1 and 2<= livecount <= 3 :
            newworld[i] = 1
        elif world[i] == 1 and livecount <= 1 :
            newworld[i] = 0
        elif world[i] == 1 and livecount > 3 :
            newworld[i] = 0
    if datetime.now().timestamp() - stamp >= 90 :
        break
    if world == newworld :
        break
    world = newworld

#sense.clear()  # clean-off
