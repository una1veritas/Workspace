import qwiic_i2c

class VEML6030:
    I2C_ADDR_DEFAULT = 0x48
    I2C_ADDR_ALT     = 0x10

    REG_CONFIG      = 0x00
    REG_ALS         = 0x04
    H_THRESH_REG    = 0x01
    L_THRESH_REG    = 0x02
    POWER_SAVE_REG  = 0x03
    AMBIENT_LIGHT_DATA_REG = 0x04
    WHITE_LIGHT_DATA_REG = 0x05
    INTERRUPT_REG   = 0x06

    NO_SHIFT               = 0x00
    INT_EN_POS             = 0x01
    PSM_POS                = 0x01
    PERS_PROT_POS          = 0x04
    INTEGTIME_POS              = 0x06
    ALS_GAIN_POS            = 11
    INT_POS                = 0xE

    ALS_GAIN_1MASK           = 3<<ALS_GAIN_POS
    THRESH_MASK            = 0x0
    INTEGTIME_1MASK              = 0xf<<INTEGTIME_POS
    PERS_PROT_MASK         = 0xFFCF
    INT_EN_MASK            = 0xFFFD
    SD_MASK                = 0xFFFE
    POW_SAVE_EN_MASK       = 0x06    # Most of this register is reserved
    POW_SAVE_MASK          = 0x01    # Most of this register is reserved
    INT_MASK               = 0xC000    

# Table of lux conversion values depending on the integration time and gain. 
# The arrays represent the all possible integration times and the index of the
# arrays represent the register's gain settings, which is directly analgous to
# their bit representations. 
    LUX_CONV = {800: [ .0036, .0072, .0288, .0576], 
                400: [ .0072, .0144, .0576, .1152],
                200: [ .0144, .0288, .1152, .2304],
                100: [ .0288, .0576, .2304, .4608],
                50: [ .0576, .1152, .4608, .9216],
                25: [ .1152, .2304, .9216, 1.8432]}

    ENABLE       = 0x01
    DISABLE      = 0x00
    SHUTDOWN     = 0x01
    POWER        = 0x00
    NO_INT       = 0x00
    INT_HIGH     = 0x01
    INT_LOW      = 0x02
    UNKNOWN_ERROR= 0xFF

    
    def __init__(self, i2c_driver=None, i2c_address = I2C_ADDR_DEFAULT):
        """
        set the instance variable _i2c
        """
        if i2c_driver is None:
            self._i2c = qwiic_i2c.getI2CDriver()
            if self._i2c is None:
                print("Unable to load I2C driver for this platform.")
                return
        else:
            self._i2c = i2c_driver
        self.address = i2c_address

    def is_connected(self):
        """
            :return: True if the device is connected, otherwise False.
            :rtype: bool

        """
        return qwiic_i2c.isDeviceConnected(self.address)

    connected = property(is_connected)
    
                
    def begin(self, confval = None):
        if not self.is_connected() :
            return False
        return True
    
    def read_word(self, reg_addr):
        val = self._i2c.readWord(self.address, reg_addr)
        return val
    
    def write_word(self, reg_addr, val):
        self._i2cwriteWord(self.address, reg_addr, val)

    def read_configuration(self):
        vals = self._i2c.readBlock(self.address, VEML6030.REG_CONFIG, 2)
        return (vals[0] << 8) | vals[1]

    def write_configuration(self, val16):
        vals = [(val16>>8) & 0xff, val16 & 0xff]
        self._i2c.writeBlock(self.address, VEML6030.REG_CONFIG, vals)
        return

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
        confval = self.read_configuration()
        confval &= (0xffff ^ VEML6030.ALS_GAIN_1MASK)
        confval |= bits<<VEML6030.ALS_GAIN_POS
        self.write_configuration(confval)
        return

    def get_gain(self):
        confval = self.read_configuration()
        bits = (confval & VEML6030.ALS_GAIN_1MASK)>>VEML6030.ALS_GAIN_POS
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
    
    gain = property(get_gain, set_gain)
    
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
        confval = self.read_configuration()
        confval &= (0xffff ^ VEML6030.INTEGTIME_1MASK)
        confval |= bits<<VEML6030.INTEGTIME_POS
        self.write_configuration(confval)
        return
        
    def get_integration_time(self):
        confval = self.read_configuration()
        confval &= VEML6030.INTEGTIME_1MASK
        bits = confval>>VEML6030.INTEGTIME_POS
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
            return VEML6030.UNKNOWN_ERROR
    
    integration_time = property(get_integration_time, set_integration_time)
    
    def read_lux(self):
        alsbits = self.read_word(VEML6030.REG_ALS)
        print("alsbits = ", hex(alsbits))
        _gain = float(self.gain)
        _integtime = int(self.integration_time)
        _convPos = {2.0: 0, 1.0: 1, 0.25: 2, 0.125: 3}.get(_gain, 1)
        _luxconv = VEML6030.LUX_CONV[_integtime][_convPos]
        lux = _luxconv * alsbits
        #print("lux = ", lux)
        if lux > 1000 :
            return self.lux_compensated(lux)
        else:
            return lux
        
    def lux_compensated(self, lux):
        compensated = (0.00000000000060135 * pow(lux, 4)) - \
        (0.0000000093924 * pow(lux, 3)) + \
        (0.000081488 * pow(lux, 2)) + (1.0023 * lux)
        return int(compensated)
        
# program start 
if __name__ == "__main__":
    from datetime import datetime
    import time
    import sys
    
    veml = VEML6030(i2c_address=VEML6030.I2C_ADDR_ALT)
    if not veml.connected :
        print("not connected.", file=sys.stderr)
    veml.begin()
    # Read the CONFIG register (2 bytes)
    config = veml.read_configuration()
    print("configuration: ", hex(config))
    veml.gain = 1/4
    print("ASL gain: ", veml.gain)
    veml.integration_time = 200
    print("ASL Integ. Time: ", veml.integration_time)
    time.sleep(1)
    #print(veml.read_word(VEML6030.REG_ALS))
    #
    # # Print out temperature every minute
    while True:
        lux = veml.read_lux()
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
              "{:.2f} lux".format(round(lux, 2)))
        time.sleep(5)
