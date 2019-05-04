'''
Created on 2019/05/03

@author: Sin Shimozono
'''
from sense_hat import SenseHat
from datetime import datetime, timedelta, timezone
import time

TZ_JST = timezone(timedelta(hours=+9), 'JST')
TZ_UTC = timezone(timedelta(hours=0), 'UTC')

WHITE = [255, 255, 255]  # White
BLACK = [0, 0, 0]  # White

ring_outer = [
    [4,0], [5,0], [6,0],
    [7,1], [7,2], [7,3], [7,4], [7,5], [7,6],
    [6,7], [5,7], [4,7], [3,7], [2,7], [1,7],
    [0,6], [0,5], [0,4], [0,3], [0,2], [0,1],
    [1,0], [2,0], [3,0]
    ]
ring_inner = [
    [4,1], [5,1], 
    [6,2], [6,3], [6,4], [6,5], 
    [5,6], [4,6], [3,6], [2,6], 
    [1,5], [1,4], [1,3], [1,2],
    [2,1], [3,1],
    ]

idx_offset = [22, 23, 0, 1, 2]
mag_sec = [
    [20, 130, 130, 20, 0], 
    [17, 123, 135, 22, 0], 
    [14, 113, 144, 25, 0], 
    [12, 111, 147, 30, 0], 
    [10, 102, 151, 33, 1], 
    [9, 95, 154, 38, 1], 
    [8, 89, 155, 44, 1], 
    [6, 84, 161, 46, 2], 
    [5, 77, 161, 53, 2], 
    [4, 71, 162, 58, 2], 
    [3, 63, 164, 63, 3], 
    [2, 60, 164, 67, 4], 
    [2, 52, 162, 77, 5], 
    [2, 46, 161, 82, 6], 
    [1, 43, 156, 90, 8], 
    [1, 37, 154, 97, 9], 
    [1, 33, 148, 106, 10], 
    [0, 30, 143, 110, 12], 
    [0, 25, 138, 119, 14], 
    [0, 22, 135, 123, 17]
    ]
mag_min = [
    [1, 158, 157, 1, 0], 
    [0, 129, 185, 4, 0], 
    [0, 98, 214, 7, 0], 
    [0, 75, 232, 12, 0], 
    [0, 50, 248, 21, 0], 
    [0, 34, 250, 34, 0], 
    [0, 20, 249, 50, 0], 
    [0, 13, 234, 72, 0], 
    [0, 8, 214, 96, 0], 
    [0, 3, 187, 127, 0]
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

def mix(col1, col2):
    mixed = list()
    mixed.append(min(col1[0] + col2[0], 255))
    mixed.append(min(col1[1] + col2[1], 255))
    mixed.append(min(col1[2] + col2[2], 255))
    return mixed
    
sense = SenseHat()
sense.clear()


ledscreen = []
fill_screen(ledscreen,BLACK)
lastdt = datetime.now(TZ_JST)
lastoctsecond = 8*lastdt.second + lastdt.microsecond // 125000
lastqmin = 4 * lastdt.minute + lastdt.second // 15
try:
    while True:
        # timestamp for this iteration
        nowdt = datetime.now(TZ_JST)
        nowqsecond = 8*nowdt.second + nowdt.microsecond // 125000
        if lastoctsecond != nowqsecond :
            lastoctsecond = nowqsecond
            nowqmin = 4 * nowdt.minute + nowdt.second // 15
            if lastqmin != nowqmin:
                lastqmin = nowqmin
            # draw outer ring
            fill_screen(ledscreen,BLACK)
            basepos = lastoctsecond // len(mag_sec)
            fracpos = lastoctsecond % len(mag_sec)
            for i in range(len(mag_sec[fracpos])):
                [x, y] = ring_outer[(basepos + idx_offset[i]) % 24]
                brt = [mag_sec[fracpos][i], mag_sec[fracpos][i], mag_sec[fracpos][i]]
                set_screen(ledscreen, x, y, brt)
            basepos = lastqmin // len(mag_min)
            fracpos = lastqmin % len(mag_min)
            for i in range(len(mag_min[fracpos])):
                [x, y] = ring_outer[(basepos + idx_offset[i]) % 24]
                brt = [mag_min[fracpos][i], mag_min[fracpos][i], 0]
                brt = mix(ledscreen[y*8+x], brt)
                set_screen(ledscreen, x, y, brt)
            # draw inner ring
            basepos = lastdt.hour // 16
            [x, y] = ring_inner[basepos]
            set_screen(ledscreen, x, y, [0, 191, 63])
            sense.set_pixels(ledscreen)
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
    
sense.clear()  # clean-off
