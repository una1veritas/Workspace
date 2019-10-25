from sense_hat import SenseHat
from datetime import datetime
import random
import time

RED = [255, 0, 0]  # Red
GREEN = [0, 255, 0]  # Green
BLUE = [0, 0, 255]  # Blue
YELLOW = [255, 255, 0] # Yellow
WHITE = [255, 255, 255]  # White
BLACK = [0, 0, 0]  # White

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
laststamp = datetime.now().timestamp()

balls = [[2, 0, 0.1]] # column, row, speed
paddle = [0,2] # left, length
try:
    while True:
        # timestamp for this iteration
        currstamp = datetime.now().timestamp()
        if currstamp - laststamp < 0.05 :
            continue
        
        # read stick events and update the posution of paddle
        for event in sense.stick.get_events():
            if (event.direction == "left" or event.direction == "right") and \
               (event.action == "pressed" or event.action == "held") :
                if event.action == "pressed" :
                    if event.direction == "left" :
                        paddle[0] = max(0, paddle[0] - 1)
                    elif event.direction == "right" :
                        paddle[0] = min(8 - paddle[1], paddle[0] + 1)
                elif event.action == "held" :
                    if event.direction == "left" :
                        paddle[0] = max(0, paddle[0] - 0.2)
                    elif event.direction == "right" :
                        paddle[0] = min(8 - paddle[1], paddle[0] + 0.2)
                sense.stick.get_events() # to make empty the event queue
                break
        # update the positions of balls 
        for b in balls:
            if b[1] < 8:
                b[1] += b[2]
        balls = [ball for ball in balls if ball[1] < 8]
        if balls[-1][1] > 2 and len(balls) < random.uniform(0, 3):
            balls.append([random.randrange(0,8), 0, random.uniform(0.03, 0.15)])
        
        # update screen
        # draw paddle
        fill_screen(ledscreen,BLACK)
        pos_x = int(paddle[0])
        for x in range(pos_x, pos_x + paddle[1]):
            set_screen(ledscreen, x, 7, YELLOW)
        # draw balls
        for b in balls:
            pos_y = int(b[1])
            set_screen(ledscreen, b[0], pos_y, WHITE)
        sense.set_pixels(ledscreen)

        laststamp = currstamp
        
except KeyboardInterrupt:
    pass
    
sense.clear()  # clean-off

