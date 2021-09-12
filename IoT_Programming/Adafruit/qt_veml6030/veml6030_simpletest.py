import time
import board
from qt_veml6030 import VEML6030
        
from datetime import datetime
import time
import sys

# program start 
i2c = board.I2C()
veml = VEML6030(i2c, VEML6030.I2C_ADDR_ALT)

# Read the CONFIG register (2 bytes)
config = veml.read_configuration()
print("configuration: ", hex(config))
veml.gain = 1
print("ASL gain: ", veml.gain)
veml.integration_time = 200
print("ASL Integ. Time: ", veml.integration_time)
config = veml.read_configuration()
print("new configuration: ", hex(config))
time.sleep(1)
#print(veml.read_word(VEML6030.REG_ALS))
#
# # Print out temperature every minute
while True:
    lux = veml.read_lux()
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
          "{:.2f} lux".format(round(lux, 2)))
    time.sleep(5)
    
