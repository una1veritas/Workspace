# STEMMA QT Vishay VEML6030 lux sensor driver
# based on Adafruit CircuitPython library
# by Sin Shimozono

#from collections import namedtuple
#from micropython import const
import adafruit_bus_device.i2c_device as i2c_device
#from adafruit_register.i2c_struct import ROUnaryStruct, UnaryStruct

#from adafruit_register.i2c_bit import RWBit, ROBit
#from adafruit_register.i2c_bits import RWBits, ROBits

class VEML6030:
    I2C_ADDR_DEFAULT    : int = 0x48
    I2C_ADDR_ALT        : int = 0x10

    REG_ALS_CONF        : int = 0x00
    REG_ALS             : int = 0x04
#    _H_THRESH_REG    = 0x01
#    _L_THRESH_REG    = 0x02
    REG_POWER_SAVING    : int = 0x03
#    _AMBIENT_LIGHT_DATA_REG = 0x04
#    _WHITE_LIGHT_DATA_REG = 0x05
#    _INTERRUPT_REG   = 0x06

# _NO_SHIFT               = 0x00
# _INT_EN_POS             = 0x01
# _PSM_POS                = 0x01
# _PERS_PROT_POS          = 0x04
    INTEGTIME_POS           : int = 0x06
    ALS_GAIN_POS        : int = 11
# _INT_POS                = 0xE
#
    ALS_GAIN_1MASK      : int = 3<<ALS_GAIN_POS
# _THRESH_MASK            = 0x0
    INTEGTIME_1MASK     : int = 0xf<<INTEGTIME_POS
# _PERS_PROT_MASK         = 0xFFCF
# _INT_EN_MASK            = 0xFFFD
    ALS_SD_1MASK        : int = 1
# _POW_SAVE_EN_MASK       = 0x06    # Most of this register is reserved
# _POW_SAVE_MASK          = 0x01    # Most of this register is reserved
# _INT_MASK               = 0xC000    

# Table of lux conversion values depending on the integration time and gain. 
# The arrays represent the all possible integration times and the index of the
# arrays represent the register's gain settings, which is directly analgous to
# their bit representations. 
    LUX_CONV   : dict = {800: [ .0036, .0072, .0288, .0576], 
                         400: [ .0072, .0144, .0576, .1152],
                         200: [ .0144, .0288, .1152, .2304],
                         100: [ .0288, .0576, .2304, .4608],
                         50: [ .0576, .1152, .4608, .9216],
                         25: [ .1152, .2304, .9216, 1.8432] }

    GAIN_BITS       : dict = {1.0: 0, 2.0: 1, 0.125: 2, 0.25: 3}
    INTEGTIME_BITS  : dict = {100: 0, 200: 1, 400: 2, 800: 3, 50: 8, 25: 12}

# _ENABLE       = 0x01
# _DISABLE      = 0x00
    ALS_SHUTDOWN    : int = 1
    ALS_POWER_ON    : int = 0
# _POWER        = 0x00
# _NO_INT       = 0x00
# _INT_HIGH     = 0x01
# _INT_LOW      = 0x02
    UNKNOWN_ERROR   : int = 0xFF
    
    def __init__(self, i2c_bus, address = I2C_ADDR_DEFAULT):
        """
        set the instance variable _i2c
        """
        self._i2c = i2c_device.I2CDevice(i2c_bus, address)
    
    def read_register(self, reg_addr):
        buf = bytearray([reg_addr, 0, 0])
        self._i2c.write_then_readinto(buf, buf, out_end=1, in_start=1)
        #print(buf)
        return buf[1]<<8 | buf[2]
        
    def write_register(self, reg_addr, val):
        self._i2c.write(bytes([reg_addr & 0xFF, (val>>8) & 0xFF, val & 0xff]))
        # print("$%02X <= 0x%02X" % (register, value))

    def read_configuration(self):
        return self.read_register(VEML6030.REG_ALS_CONF)

    def write_configuration(self, val16):
        self.write_register(VEML6030.REG_ALS_CONF, val16)
    
    configuration = property(read_configuration, write_configuration)

    def get_gain(self):
        bits = (self.configuration & VEML6030.ALS_GAIN_1MASK)>>VEML6030.ALS_GAIN_POS
        for (key, val) in VEML6030.GAIN_BITS.items():
            if val == bits :
                return key
    
    def set_gain(self, gainVal):
        bits = VEML6030.GAIN_BITS.get(gainVal, 0)
        confval = self.configuration
        confval &= (0xffff ^ VEML6030.ALS_GAIN_1MASK)
        confval |= bits<< VEML6030.ALS_GAIN_POS
        self.configuration = confval
        return

    gain = property(get_gain, set_gain)
    
    def get_integration_time(self):
        confval = self.configuration
        confval &= VEML6030.INTEGTIME_1MASK
        bits = confval>> VEML6030.INTEGTIME_POS
        #print(hex(confval), bin(bits))
        for (key, val) in VEML6030.INTEGTIME_BITS.items():
            if val == bits :
                return key
    
    def set_integration_time(self, time = 100):
        bits = VEML6030.INTEGTIME_BITS.get(time, 0)
        confval = self.configuration
        confval &= (0xffff ^ VEML6030.INTEGTIME_1MASK)
        confval |= bits<< VEML6030.INTEGTIME_POS
        self.configuration = confval
        return
        
    integration_time = property(get_integration_time, set_integration_time)
    
    def set_gain_integtime(self, g, t):
        gain_bits = VEML6030.GAIN_BITS.get(g, VEML6030.GAIN_BITS[1])
        time_bits = VEML6030.INTEGTIME_BITS.get(t, VEML6030.INTEGTIME_BITS[100])
        confval = self.configuration
        confval &= (0xffff ^ (VEML6030.ALS_GAIN_1MASK | VEML6030.INTEGTIME_1MASK))
        confval |= (gain_bits<< VEML6030.ALS_GAIN_POS)|(time_bits<<VEML6030.INTEGTIME_POS)
        self.configuration = confval
        
    def get_gain_integtime(self):
        confval = self.configuration
        integtime_bits = (confval & VEML6030.INTEGTIME_1MASK)>> VEML6030.INTEGTIME_POS
        gain_bits = (confval & VEML6030.ALS_GAIN_1MASK)>>VEML6030.ALS_GAIN_POS
        #print(hex(confval), bin(bits))
        pair = [0, 0]
        for (key, val) in VEML6030.GAIN_BITS.items():
            if val == gain_bits :
                pair[0] = key
                break
        for (key, val) in VEML6030.INTEGTIME_BITS.items():
            if val == integtime_bits :
                pair[1] = key
                break
        return tuple(pair)
            
    def shutdown(self):
        confval = self.configuration
        confval &= 0xffff ^ VEML6030.ALS_SD_1MASK
        confval |= VEML6030.ALS_SHUTDOWN
        self.configuration = confval
        
    def power_on(self):
        confval = self.configuration
        confval &= 0xffff ^ VEML6030.ALS_SD_1MASK
        confval |= VEML6030.ALS_POWER_ON
        self.configuration = confval
        
    def read_lux(self):
        alsbits = self.read_register(VEML6030.REG_ALS)
        #print("alsbits = ", hex(alsbits))
        g = float(self.gain)
        it = int(self.integration_time)
        conv = {2.0: 0, 1.0: 1, 0.25: 2, 0.125: 3}.get(g, 1)
        lux = VEML6030.LUX_CONV[it][conv] * alsbits
        print("als = ", hex(alsbits), "lux = ", lux, "conv = ", conv, "integtime = ", it, "gain = ", g)
        if lux > 1000 :
            return self.lux_compensated(lux)
        else:
            return lux

    lux = property(read_lux)
        
    def lux_compensated(self, lux):
        compensated = (0.00000000000060135 * pow(lux, 4)) - \
        (0.0000000093924 * pow(lux, 3)) + \
        (0.000081488 * pow(lux, 2)) + (1.0023 * lux)
        return int(compensated)
