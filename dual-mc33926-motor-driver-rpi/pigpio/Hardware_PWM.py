'''
Created on 2025/02/14

@author: sin
'''
import pigpio
import time

gpio_pin0 = 18
gpio_pin1 = 19
# 12  PWM channel 0  All models but A and B
# 13  PWM channel 1  All models but A and B

pi = pigpio.pi()
pi.set_mode(gpio_pin0, pigpio.OUTPUT)
pi.set_mode(gpio_pin1, pigpio.OUTPUT)

# GPIO18: 2Hz、duty比0.5
pi.hardware_PWM(gpio_pin0, 2, 500000)
# GPIO19: 8Hz、duty比0.1
pi.hardware_PWM(gpio_pin1, 8, 100000)

time.sleep(5)

pi.set_mode(gpio_pin0, pigpio.INPUT)
pi.set_mode(gpio_pin1, pigpio.INPUT)
pi.stop()
