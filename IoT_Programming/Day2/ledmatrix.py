from sense_hat import SenseHat
import time
import random
from datetime import datetime

def fill(screen, color):
    screen.clear()
    for i in range(8*8):
        screen.append(color)
    return

def set_xy(screen, x, y, color):
    if 0 <= x < 8 and 0 <= y < 8 :
        screen[y*8 + x] = color
    return

sense = SenseHat()
sense.clear()

# flowing text
wait = 0.07
bg = [96, 96, 64] # R, G, B
fg = [192, 192, 255]
sense.show_message('Hello!', wait, fg, bg)

for angle in [0, 90, 180, 270]:
    sense.set_rotation(angle)
    sense.show_message(str(angle))
sense.set_rotation(0)

for a_letter in "Get the key!!!":
    sense.clear()
    time.sleep(0.1)
    sense.show_letter(a_letter)
    time.sleep(0.3)
time.sleep(1)
sense.clear()

yellow = [255, 255, 0]  # Yellow
blue = [0, 0, 255]  # Blue

packman = [[
0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 0, 0,
0, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 1, 1, 0,
0, 0, 1, 1, 1, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0,
],
    [
0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 0, 0,
0, 1, 1, 1, 1, 1, 1, 0,
0, 1, 1, 1, 1, 0, 0, 0,
0, 1, 1, 0, 0, 0, 0, 0,
0, 1, 1, 1, 1, 1, 1, 0,
0, 0, 1, 1, 1, 1, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0,
]]

#1/0 のリストをカラーに置き換える
for i in range(0,len(packman)):
    pattern = packman[i]
    packman[i] = [blue if c == 0 else yellow for c in pattern]

for i in range(0,5):
    sense.set_pixels(packman[i%2])
    time.sleep(0.6)

sense.clear()

world = [0 for i in range(64)]
for i in [4, 12, 18, 19, 21, 22, 23, 28, 36]:
    world[i] = 1
laststamp = datetime.now().timestamp()

colors = [[0,0,0], [100,100,100]]

while True:
    newworld = world.copy()
    for i in range(0,64):
        livecount = 0
        for d in [-9, -8, -7, -1, 1, 7, 8, 9]:
            if 0 <= i+d < 64 :
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
    world = newworld
    scrn = [colors[0] if c == 0 else colors[1] for c in world]
    sense.set_pixels(scrn)
    wait(1)

sense.clear()
sense.show_message('Bye!')

sense.clear()  # clean-off
