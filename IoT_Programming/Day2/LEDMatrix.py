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

X = [255, 255, 0]  # Red
O = [0, 0, 255]  # White

packman = [[
O, O, O, O, O, O, O, O,
O, O, X, X, X, X, O, O,
O, X, X, X, X, X, X, O,
O, X, X, X, X, X, X, O,
O, X, X, X, X, X, X, O,
O, X, X, X, X, X, X, O,
O, O, X, X, X, X, O, O,
O, O, O, O, O, O, O, O,
],
    [
O, O, O, O, O, O, O, O,
O, O, X, X, X, X, O, O,
O, X, X, X, X, X, X, O,
O, X, X, X, X, O, O, O,
O, X, X, O, O, O, O, O,
O, X, X, X, X, X, X, O,
O, O, X, X, X, X, O, O,
O, O, O, O, O, O, O, O,
]]

for i in range(0,5):
    sense.set_pixels(packman[i%2])
    time.sleep(0.6)

sense.clear()

dots = []
scrn = []
laststamp = datetime.now().timestamp()
endstamp = laststamp + 10
while True:
    newstamp = datetime.now().timestamp()
    if newstamp > endstamp :
        break
    if newstamp - laststamp > 0.05:
        laststamp = newstamp
        if len(dots) == 0 :
            x = random.randrange(0,8)
            y = 0
            dots.append([x, y])
        else:
            dots = [ [d[0], d[1]+1] for d in dots if d[1] < 8]
        fill(scrn,[0,0,0])
        for a_dot in dots:
            set_xy(scrn, a_dot[0], a_dot[1], [192, 192, 64])
        sense.set_pixels(scrn)
        
sense.clear()
sense.show_message('Bye!')

sense.clear()  # clean-off
