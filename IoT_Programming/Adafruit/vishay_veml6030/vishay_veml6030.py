import time
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
    INTEG_POS           : int = 0x06
    ALS_GAIN_POS        : int = 11
# _INT_POS                = 0xE
#
    ALS_GAIN_MASK       : int = 3<<ALS_GAIN_POS
# _THRESH_MASK            = 0x0
    INTEG_MASK          : int = 0xf<<INTEG_POS
# _PERS_PROT_MASK         = 0xFFCF
# _INT_EN_MASK            = 0xFFFD
    ALS_SD_MASK         : int = 1
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

# _ENABLE       = 0x01
# _DISABLE      = 0x00
    ALS_SHUTDOWN        : int = 1
    ALS_POWER_ON        : int = 0
# _POWER        = 0x00
# _NO_INT       = 0x00
# _INT_HIGH     = 0x01
# _INT_LOW      = 0x02
    UNKNOWN_ERROR      : int = 0xFF

    
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
        bits = (self.configuration & VEML6030.ALS_GAIN_MASK)>>VEML6030.ALS_GAIN_POS
        if bits == 0 :
            return 1.00
        elif bits == 1 : 
            return 2.00
        elif bits == 2 :
            return 0.125
        elif bits == 3 :
            return 0.25
        else :
            return VEML6030.UNKNOWN_ERROR
    
    def set_gain(self, gainVal):
        if (gainVal == 1.00) :
            bits = 0 
        elif (gainVal == 2.00) :
            bits = 1
        elif (gainVal == .125) :
            bits = 2
        elif (gainVal == .25) :
            bits = 3
        else:
            return
        confval = self.configuration
        confval &= (0xffff ^ VEML6030.ALS_GAIN_MASK)
        confval |= bits<< VEML6030.ALS_GAIN_POS
        self.configuration = confval
        return

    gain = property(get_gain, set_gain)
    
    def get_integration_time(self):
        confval = self.configuration
        confval &= VEML6030.INTEG_MASK
        bits = confval>> VEML6030.INTEG_POS
        #print(hex(confval), bin(bits))
        if bits == 0 :
            return 100
        elif bits == 1 : 
            return 200
        elif bits == 2 :
            return 400
        elif bits == 3 :
            return 800
        elif bits == 8 :
            return 50
        elif bits == 12 :
            return 25
        else:
            return VEML6030._UNKNOWN_ERROR
    
    def set_integration_time(self, time = 100):
        if (time == 100) :
            bits = 0 
        elif (time == 200) :
            bits = 1
        elif (time == 400) :
            bits = 2
        elif (time == 800) :
            bits = 3
        elif (time == 50) :
            bits = 8
        elif (time == 25) :
            bits = 12
        else:
            return
        confval = self.configuration
        confval &= (0xffff ^ VEML6030.INTEG_MASK)
        confval |= bits<< VEML6030.INTEG_POS
        self.configuration = confval
        return
        
    integration_time = property(get_integration_time, set_integration_time)
    
    def shutdown(self):
        confval = self.configuration
        confval &= 0xffff ^ VEML6030.ALS_SD_MASK
        confval |= VEML6030.ALS_SHUTDOWN
        self.configuration = confval
        
    def power_on(self):
        confval = self.configuration
        confval &= 0xffff ^ VEML6030.ALS_SD_MASK
        confval |= VEML6030.ALS_POWER_ON
        self.configuration = confval
        
    def read_lux(self):
        alsbits = self.read_register(VEML6030.REG_ALS)
        #print("alsbits = ", hex(alsbits))
        gain = float(self.gain)
        integtime = int(self.integration_time)
        conv = {2.0: 0, 1.0: 1, 0.25: 2, 0.125: 3}.get(gain, 1)
        lux = VEML6030.LUX_CONV[integtime][conv]
        lux = lux * alsbits
        #print("lux = ", lux)
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
