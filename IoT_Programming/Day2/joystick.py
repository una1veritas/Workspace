#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sense_hat import SenseHat
import time

sense = SenseHat()
sense.clear()

while True: # イベントループ
    for event in sense.stick.get_events():
        if event.direction == "up" and event.action == "pressed" :
            for c in "up pressed." :
                sense.show_letter(c)
                time.sleep(0.2)
                sense.clear()
                time.sleep(0.05)
            sense.stick.get_events()  # show_message の実行時間中に発生したイベントを読み捨てる
            break
    time.sleep(0.2)


    
        
