'''
Created on 2025/03/15

@author: sin
'''
import serial
import time
import serial.tools.list_ports
from pickle import NONE

def read_message(ser):
    while ser.in_waiting > 0:
        message = ser.readline().decode().strip()
        print("Received:", message)

if __name__ == '__main__':
    '''automatically select serial port through USB/micro:bit'''
    ports = serial.tools.list_ports.comports()
    ser_port = None
    port_desc = ''
    for port in ports:
        if 'BBC micro:bit' in port.description:
            ser_port = port.device
            port_desc = port.description
            break
    print(f'selected {ser_port}, {port_desc}')
    # Replace 'COM3' with the correct port for your micro:bit
    ser = serial.Serial(ser_port, 115200)
    
    try:
        while True:
            read_message(ser)
            #time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()
