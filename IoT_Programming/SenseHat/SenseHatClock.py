'''
Created on 2019/05/03

@author: Sin Shimozono
'''
from sense_hat import SenseHat
from datetime import datetime
import random
import time

WHITE = [255, 255, 255]  # White
BLACK = [0, 0, 0]  # White

ringpos = [
    [4,0], [5,0], [6,0],
    [7,1], [7,2], [7,3], [7,4], [7,5], [7,6],
    [6,7], [5,7], [4,7], [3,7], [2,7], [1,7],
    [0,6], [0,5], [0,4], [0,3], [0,2], [0,1],
    [1,0], [2,0], [3,0]
    ]

def fill_screen(screen, color):
    screen.clear()
    for i in range(8*8):
        screen.append(color)
    return

def set_screen(screen, column, row, color):
    if 0 <= column < 8 and 0 <= row < 8 :
        screen[row*8 + column] = color
    return

sense = SenseHat()
sense.clear()


ledscreen = []
fill_screen(ledscreen,BLACK)
lastdt = datetime.now()

try:
    while True:
        # timestamp for this iteration
        currentdt = datetime.now()
        if lastdt.second != currentdt.second :
            lastdt = currentdt
            secfrac = lastdt.second % 5
            sec24th = lastdt.second*2/5
            fill_screen(ledscreen,BLACK)
            print(int(sec24th))
            if secfrac == 0 :
                [x, y] = ringpos[int(sec24th + 23) % 24]
                set_screen(ledscreen,x,y,[127,127,127])
                [x, y] = ringpos[int(sec24th)]
                set_screen(ledscreen,x,y,[127,127,127])
            elif secfrac == 1 :
                [x, y] = ringpos[int(sec24th + 23) % 24]
                set_screen(ledscreen,x,y,[53,53,53])
                [x, y] = ringpos[int(sec24th)]
                set_screen(ledscreen,x,y,[203,203,203])
            elif secfrac == 2:
                [x, y] = ringpos[int(sec24th)]
                set_screen(ledscreen,x,y,[203,203,203])
                [x, y] = ringpos[int(sec24th + 1) % 24]
                set_screen(ledscreen,x,y,[53,53,53])
            elif secfrac == 3 :
                [x, y] = ringpos[int(sec24th + 23) % 24]
                set_screen(ledscreen,x,y,[53,53,53])
                [x, y] = ringpos[int(sec24th)]
                set_screen(ledscreen,x,y,[203,203,203])
            elif secfrac == 4 :
                [x, y] = ringpos[int(sec24th)]
                set_screen(ledscreen,x,y,[203,203,203])
                [x, y] = ringpos[int(sec24th + 1) % 24]
                set_screen(ledscreen,x,y,[53,53,53])
            #[x, y] = ringpos[int(lastdt.second*4/10) % 24]
            #set_screen(ledscreen,x,y,WHITE)
            sense.set_pixels(ledscreen)
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
    
sense.clear()  # clean-off
