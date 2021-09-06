import qwiic_i2c

class TMP117:
    REG_TEMP_RESULT = 0x00
    REG_CONFIG      = 0x01
    
    CONFIG_DATA_READY       = 0x0001 << 13
    CONFIG_AVG_NO_AVERAGE   = 0
    CONFIG_AVG_8_AVERAGE    = 0x0001 << 5
    config_avg_32_average   = 0x0002 << 5
    config_avg_64_average   = 0x0003 << 5
    CONFIG_AVG_MASK         = 0x0003 << 5
    config_mod_cont         = 0
    config_mod_shutdown     = 0x0001<<10
    config_mod_one_shot     = 0x0003<<10
    CONFIG_MOD_MASK         = 0x0003<<10
    config_conv_000         = 0x0000<<7
    config_conv_001         = 0x0001<<7
    config_conv_010         = 0x0002<<7
    config_conv_011         = 0x0003<<7
    config_conv_100         = 0x0004<<7
    config_conv_101         = 0x0005<<7
    config_conv_110         = 0x0006<<7
    config_conv_111         = 0x0007<<7
    CONFIG_CONV_MASK        = 0x0007<<7
    CONFIG_SOFT_RESET       = 0x0001<<1

    config_factorysetting   = 0x0220
    resolution              = 0.0078125 # 1/128 Centigrade
    
    def __init__(self, i2c_driver=None):
        """
        set the instance variable _i2c
        """
        if i2c_driver is None:
            self._i2c = qwiic_i2c.getI2CDriver()
            if self._i2c is None:
                print("Unable to load I2C driver for this platform.")
        else:
            self._i2c = i2c_driver
        self.address = 0x48

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
        if confval != None:
            self.write_configuration(confval)
        return True
    
    # def read(self, reg_addr):
    #     vals = self._i2c.readWord(self.i2c_address, reg_addr, 2)
    #     return vals
    #
    # def write(self, reg_addr, vals):
    #     self.i2c_bus.write_block_data(self.i2c_address,reg_addr,vals)
        
    def read_temperature(self):
        val = self._i2c.readBlock(self.address, TMP117.REG_TEMP_RESULT, 2)
        t = (val[0] << 8) | val[1]
        if (t & 0x8000) != 0 :
            t = -((t^0xffff)+1)
        t *= self.resolution
        return t
        
    def write_configuration(self, val16bit):
        vals = [(val16bit>>8)&0xff, val16bit & 0xff]
        self._i2c.writeBlock(self.address, TMP117.REG_CONFIG, vals)
        return

    def read_configuration(self):
        vals = self._i2c.readBlock(self.address, TMP117.REG_CONFIG, 2)
        return (vals[0] << 8) | vals[1]

    def is_ready(self):
        return (self.read_configuration() & TMP117.CONFIG_DATA_READY) != 0

    ready = property(is_ready)

    def soft_reset(self):
        val16 = self.read_configuration()
        val16 |= TMP117.CONFIG_SOFT_RESET
        self.write_configuration(val16)

    def get_mode(self):
        val16 = self.read_configuration()
        return (val16 & TMP117.CONFIG_MOD_MASK)>>10
        
    def set_mode(self, mode = 0):
        val16 = self.read_configuration()
        val16 &= TMP117.CONFIG_MOD_MASK
        val16 |= (mode & 0b11)<<10
        self.write_configuration(val16)
     
# program start 
if __name__ == "__main__":
    from datetime import datetime
    import time
    import sys
    
    tmp117 = TMP117()
    if not tmp117.connected :
        print("not connected.", file=sys.stderr)
    tmp117.begin()
    # Read the CONFIG register (2 bytes)
    #config = tmp117.read_configuration()
    #print("configuration: ", hex(config))
    # Write 4 Hz sampling back to reg_config
    #config &= ~(TMP117.config_conv_mask | TMP117.CONFIG_AVG_MASK | TMP117.config_mod_mask)
    #config |= (TMP117.config_conv_011 | TMP117.CONFIG_AVG_8_AVERAGE | TMP117.config_mod_cont)
    #print("set configuration: ", hex(config))
    tmp117.write_configuration(0x01a0)
    config = tmp117.read_configuration()
    print("configuration: ", hex(config))
    tmp117.soft_reset()
    while not tmp117.ready :
        pass
    
    # Print out temperature every minute
    while True:
        temperature = tmp117.read_temperature()
        print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
              "{:.2f}C".format(round(temperature, 2)))
        time.sleep(5)
