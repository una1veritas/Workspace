# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2021 ladyada for Adafruit
#
# SPDX-License-Identifier: MIT
"""
`adafruit_sht4x`
================================================================================

Python library for Sensirion SHT4x temperature and humidity sensors

* Author(s): ladyada

Implementation Notes
--------------------

**Hardware:**

* `Adafruit SHT40 Temperature & Humidity Sensor
  <https://www.adafruit.com/product/4885>`_ (Product ID: 4885)

**Software and Dependencies:**

* Adafruit CircuitPython firmware for the supported boards:
  https://circuitpython.org/downloads

* Adafruit's Bus Device library:
  https://github.com/adafruit/Adafruit_CircuitPython_BusDevice

"""

import qwiic_i2c
import time

__version__ = "0.8"

# class CV:
#     """struct helper"""
#
#     @classmethod
#     def add_values(cls, value_tuples):
#         """Add CV values to the class"""
#         cls.string = {}
#         cls.delay = {}
#
#         for value_tuple in value_tuples:
#             name, value, string, delay = value_tuple
#             setattr(cls, name, value)
#             cls.string[value] = string
#             cls.delay[value] = delay
#
#     @classmethod
#     def is_valid(cls, value):
#         """Validate that a given value is a member"""
#         return value in cls.string

class SHT4x:
    """
    A driver for the SHT4x temperature and humidity sensor.

    :param ~busio.I2C i2c_bus: The I2C bus the SHT4x is connected to.
    :param int address: The I2C device address. Default is :const:`0x44`


    **Quickstart: Importing and using the SHT4x temperature and humidity sensor**

        Here is an example of using the :class:`SHT4x`.
        First you will need to import the libraries to use the sensor

        .. code-block:: python

            import board
            import adafruit_sht4x

        Once this is done you can define your `board.I2C` object and define your sensor object

        .. code-block:: python

            i2c = board.I2C()   # uses board.SCL and board.SDA
            sht = adafruit_sht4x.SHT4x(i2c)

        You can now make some initial settings on the sensor

        .. code-block:: python

            sht.mode = adafruit_sht4x.Mode.NOHEAT_HIGHPRECISION

        Now you have access to the temperature and humidity using the :attr:`measurements`.
        It will return a tuple with the :attr:`temperature` and :attr:`relative_humidity`
        measurements


        .. code-block:: python

            temperature, relative_humidity = sht.measurements

    """
    _SHT4X_DEFAULT_ADDR = 0x44  # SHT4X I2C Address
    _SHT4X_READSERIAL = 0x89  # Read Out of Serial Register
    _SHT4X_SOFTRESET = 0x94  # Soft Reset

    MODES = {"NOHEAT_HIGHPRECISION": (0xFD, "No heater, high precision", 0.01),
             "NOHEAT_MEDPRECISION": (0xF6, "No heater, med precision", 0.005),
             "NOHEAT_LOWPRECISION": (0xE0, "No heater, low precision", 0.002),
             "HIGHHEAT_1S": (0x39, "High heat, 1 second", 1.1),
             "HIGHHEAT_100MS": (0x32, "High heat, 0.1 second", 0.11),
             "MEDHEAT_1S": (0x2F, "Med heat, 1 second", 1.1),
             "MEDHEAT_100MS": (0x24, "Med heat, 0.1 second", 0.11),
             "LOWHEAT_1S": (0x1E, "Low heat, 1 second", 1.1),
             "LOWHEAT_100MS": (0x15, "Low heat, 0.1 second", 0.11),
             }

    def __init__(self, i2c_driver=None, i2c_address=_SHT4X_DEFAULT_ADDR):
        if i2c_driver is None:
            self._i2c = qwiic_i2c.getI2CDriver()
            if self._i2c is None:
                print("Unable to load I2C driver for this platform.")
        else:
            self._i2c = i2c_driver
        self.address = i2c_address
        self.mode = 0xfd #SHT4x.MODES["NOHEAT_HIGHPRECISION"]  # pylint: disable=no-member
    
    def is_connected(self):
        """
            :return: True if the device is connected, otherwise False.
            :rtype: bool

        """
        return qwiic_i2c.isDeviceConnected(self.address)

    connected = property(is_connected)

    def begin(self, command_mode = 0xfd):
        self.reset()
        self.mode = command_mode
        return True
    
    def reset(self):
        """Perform a soft reset of the sensor, resetting all settings to their power-on defaults"""
        self._i2c.writeCommand(self.address, 0x94)
        time.sleep(0.001)

    def get_serial_number(self):
        """The unique 32-bit serial number"""
        print(type(self._i2c._i2cbus))
        
        self._i2c.writeCommand(self.address, SHT4x._SHT4X_READSERIAL)
        vals = bytearray(6)
        for i in range(6):
            vals[i] = self._i2c.readByte(self.address)
        ser1 = vals[0:2]
        ser1_crc = vals[2]
        ser2 = vals[3:5]
        ser2_crc = vals[5]
        
        # check CRC of bytes
        if ser1_crc != self._crc8(ser1) or ser2_crc != self._crc8(ser2):
            raise RuntimeError("Invalid CRC calculated")
        
        serial = (vals[0] << 24) + (vals[1] << 16) + (vals[0] << 8) + vals[1]
        return serial
    
    serial = property(get_serial_number)

#    @property
    def get_mode(self):
        """The current sensor reading mode (heater and precision)"""
        return self.mode

#    @mode.setter
    def set_mode(self, new_mode):
        self.mode = new_mode

# #    @property
#     def relative_humidity(self):
#         """The current relative humidity in % rH. This is a value from 0-100%."""
#         return self.measurements[1]
#
# #    @property
#     def temperature(self):
#         """The current temperature in degrees Celsius"""
#         return self.measurements[0]

#    @property
    def measure(self):
        """both `temperature` and `relative_humidity`, read simultaneously"""
        result = self._i2c.writeCommand(self.address, 0xfd)
        time.sleep(0.01)
        result = 0
        for i in range(6) :
            result = (result<<8) | self._i2c.readByte(self.address)
        return result

        # # separate the read data
        # temp_data = self._buffer[0:2]
        # temp_crc = self._buffer[2]
        # humidity_data = self._buffer[3:5]
        # humidity_crc = self._buffer[5]
        #
        # # check CRC of bytes
        # if temp_crc != self._crc8(temp_data) or humidity_crc != self._crc8(
        #     humidity_data
        # ):
        #     raise RuntimeError("Invalid CRC calculated")
        #
        # # decode data into human values:
        # # convert bytes into 16-bit signed integer
        # # convert the LSB value to a human value according to the datasheet
        # temperature = struct.unpack_from(">H", temp_data)[0]
        # temperature = -45.0 + 175.0 * temperature / 65535.0
        #
        # # repeat above steps for humidity data
        # humidity = struct.unpack_from(">H", humidity_data)[0]
        # humidity = -6.0 + 125.0 * humidity / 65535.0
        # humidity = max(min(humidity, 100), 0)
        #
        # return (temperature, humidity)

    ## CRC-8 formula from page 14 of SHTC3 datasheet
    # https://media.digikey.com/pdf/Data%20Sheets/Sensirion%20PDFs/HT_DS_SHTC3_D1.pdf
    # Test data [0xBE, 0xEF] should yield 0x92

    @staticmethod
    def _crc8(buffer):
        """verify the crc8 checksum"""
        crc = 0xFF
        for byte in buffer:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x31
                else:
                    crc = crc << 1
        return crc & 0xFF  # return the bottom 8 bits

# program start 
if __name__ == "__main__":
    from datetime import datetime
    import time
    import sys
    
    sht4x = SHT4x()
    if not sht4x.connected :
        print("not connected.", file=sys.stderr)
    sht4x.begin()
#    sht4x.reset()
    print("serial: ", hex(sht4x.get_serial_number()))
    # Read the CONFIG register (2 bytes)
    #config = tmp117.read_configuration()
    #print("configuration: ", hex(config))
    # Write 4 Hz sampling back to reg_config
    #config &= ~(TMP117.config_conv_mask | TMP117.CONFIG_AVG_MASK | TMP117.config_mod_mask)
    #config |= (TMP117.config_conv_011 | TMP117.CONFIG_AVG_8_AVERAGE | TMP117.config_mod_cont)
    #print("set configuration: ", hex(config))
    # tmp117.write_configuration(0x01a0)
    # config = tmp117.read_configuration()
    # print("configuration: ", hex(config))
    # tmp117.soft_reset()
    # while not tmp117.ready :
    #     pass
    # Print out temperature every minute
    while True:
        print("measurement: ", sht4x.measure())
        time.sleep(5)
