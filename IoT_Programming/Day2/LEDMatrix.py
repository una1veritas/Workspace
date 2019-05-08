from sense_hat import SenseHat
import time

sense = SenseHat()
sense.clear()

# flowing text
sense.show_message('Hello!')

sense.show_message('angle')
for angle in [0, 90, 180, 270]:
    sense.set_rotation(angle)
    sense.show_message(str(angle))
sense.set_rotation(0)

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

ball = [3,0]
direction = [1,1]
for i in range(50):
    sense.set_pixel(ball[0], ball[1], 0, 0, 0)
    if ball[0] + direction[0] < 0 or 7 < ball[0] + direction[0]:
        direction[0] = -direction[0]
    if ball[1] + direction[1] < 0 or 7 < ball[1] + direction[1]:
        direction[1] = -direction[1]
    ball[0] += direction[0]
    ball[1] += direction[1]
    sense.set_pixel(ball[0], ball[1], 255, 255, 255)
    time.sleep(0.1)

wait = 0.05
bg = [96, 96, 64] # R, G, B
fg = [192, 192, 255]
sense.show_message('Good bye!', wait, fg, bg)

sense.clear()  # clean-off

