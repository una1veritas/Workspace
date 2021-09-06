import time
from datetime import datetime
import smbus
from pip._internal import resolution

class TMP117:
    i2c_address = 0x48
    REG_TEMP_RESULT = 0x00
    REG_CONFIG= 0x01
    i2c_bus = None
    
    config_factorysetting = 0x0220
    resolution = 0.0078125 # 1/128 in Centigrade
    
    def __init__(self, i2c_bus=None):
        if i2c_bus == None :
            print("I2C bus is not specified.")
        
    def begin(self, confval = config_factorysetting):
        self.write_configuration(confval)
    
    def read(self, reg_addr):
        vals = self.i2c_bus.read_i2c_block_data(self.i2c_addr, reg_addr, 2)
        return vals
    
    def write(self, reg_addr, vals):
        self.i2c_bus.write_block_data(self.i2c_addr,reg_addr,vals)
        
    def read_temperature(self):
        val = self.read(self.REG_TEMP_RESULT)
        temp = (val[0] << 8) | val[1]
        if (temp & 0x8000) != 0 :
            temp = -((temp^0xffff)+1)
        temp *= self.resolution
        return temp
    
    def write_configuration(self, val16bit):
        vals = [(val16bit>>8)&0xff, val16bit & 0xff]
        self.i2c_bus.write_i2c_block_data(self.i2c_addr, self.REG_CONFIG, vals)
        return

    def read_configuration(self):
        vals = self.i2c_bus.read_i2c_block_data(self.i2c_addr, self.REG_CONFIG, 2)
        return vals[0]<<8 | vals[1]

# program start 
i2c_channel = 1

# Initialize I2C (SMBus)
bus = smbus.SMBus(i2c_channel)

tmp117 = TMP117(bus)
tmp117.begin(0x01a0)

# Read the CONFIG register (2 bytes)
config = tmp117.read_configuration()
#print("configuration: ", hex(config))
# Write 4 Hz sampling back to reg_config
#config = (config & 0b1111110001111111) | ((3 & 0b111)<< 7)
#tmp117.write_configuration(config)
#config = tmp117.read_configuration()
print("configuration: ", hex(config))

# Print out temperature every minute
last = datetime.now()
while True:
    dtnow = datetime.now()
    if last.minute != dtnow.minute and dtnow.second == 0 :
        temperature = tmp117.read_temperature()
        print(dtnow.strftime("%Y/%m/%d %H:%M:%S"),
              "{:.2f}C".format(round(temperature, 2)))
        last = dtnow
    time.sleep(1)
