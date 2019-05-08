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
    [4,0], [5,0], [6,1], [7,2], [7,3], [7,4], [7,5], [6,6], [5,7], [4,7], 
    [3,7], [2,7], [1,6], [0,5], [0,4], [0,3], [0,2], [1,1], [2,0], [3,0]
    ]
ring_inner = [
    [4,1], [5,1], 
    [6,2], [6,3], [6,4], [6,5], 
    [5,6], [4,6], [3,6], [2,6], 
    [1,5], [1,4], [1,3], [1,2],
    [2,1], [3,1],
    ]

idx_offset = [18, 19, 0, 1, 2]
mag_sec = [
    [19, 139, 141, 19, 0], 
    [14, 130, 151, 22, 0], 
    [9, 117, 161, 31, 0], 
    [7, 103, 169, 37, 1], 
    [5, 92, 174, 46, 1], 
    [4, 76, 182, 55, 1], 
    [2, 68, 178, 66, 2], 
    [1, 55, 179, 78, 4], 
    [1, 48, 175, 88, 5], 
    [0, 37, 171, 101, 7], 
    [0, 31, 161, 117, 9], 
    [0, 22, 154, 128, 14]
    ]
mag_min = [
    [2, 156, 158, 2, 0], 
    [1, 132, 181, 4, 0], 
    [0, 108, 202, 7, 0], 
    [0, 86, 219, 12, 0], 
    [0, 68, 232, 18, 0], 
    [0, 51, 242, 26, 0], 
    [0, 37, 245, 36, 0], 
    [0, 26, 240, 52, 0], 
    [0, 18, 231, 70, 0], 
    [0, 11, 218, 89, 0], 
    [0, 7, 201, 109, 0], 
    [0, 4, 178, 135, 1]
    ]
mag_hour = [
    [139, 139, 0], 
    [85, 193, 1], 
    [42, 230, 5], 
    [18, 242, 18], 
    [5, 230, 42], 
    [1, 196, 85],
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
lastquartersec = 61
try:
    while True:
        # timestamp for this iteration
        currentdt = datetime.now(TZ_JST)
        currentquatersec = 4*currentdt.second + currentdt.microsecond // 250000
        if lastquartersec != currentquatersec :
            lastquartersec = currentquatersec
            currentquartermin = 4 * currentdt.minute + currentdt.second // 15
            # draw outer ring
            fill_screen(ledscreen,BLACK)
            basepos = lastquartersec // len(mag_sec)
            fracpos = lastquartersec % len(mag_sec)
            for i in range(len(mag_sec[fracpos])):
                [x, y] = ring_outer[(basepos + idx_offset[i]) % len(ring_outer)]
                brt = [mag_sec[fracpos][i], mag_sec[fracpos][i], mag_sec[fracpos][i]]
                set_screen(ledscreen, x, y, brt)
            basepos = currentquartermin // len(mag_min)
            fracpos = currentquartermin % len(mag_min)
            for i in range(len(mag_min[fracpos])):
                [x, y] = ring_outer[(basepos + idx_offset[i]) % len(ring_outer)]
                brt = [mag_min[fracpos][i], mag_min[fracpos][i], 0]
                brt = mix(ledscreen[y*8+x], brt)
                set_screen(ledscreen, x, y, brt)
            # draw inner ring
            basepos = (4 * currentdt.hour + currentdt.minute // 15) // len(mag_hour)
            fracpos = (4 * currentdt.hour + currentdt.minute // 15) % len(mag_hour)
            #print(basepos, fracpos)
            for i in range(3):
                [x, y] = ring_inner[(basepos + [15, 0, 1][i]) % len(ring_inner)]
                #print(x,y)
                brt = [0, mag_hour[fracpos][i]*3//4, mag_hour[fracpos][i]]
                set_screen(ledscreen, x, y, brt)
            sense.set_pixels(ledscreen)
        time.sleep(0.05)
except KeyboardInterrupt:
    pass
    
sense.clear()  # clean-off
