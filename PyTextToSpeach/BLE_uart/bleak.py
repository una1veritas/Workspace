'''
Created on 2025/03/15

@author: sin
'''

import asyncio
from bleak import BleakClient, BleakScanner

async def main():
    # Replace with your micro:bit's MAC address
    MAC_ADDRESS = "your_microbit_mac_address"
    
    # Define the UART service and characteristic UUIDs
    UART_SERVICE_UUID = "00001100-0000-1000-8000-00805F9B0131"
    RX_CHARACTERISTIC_UUID = "00001101-0000-1000-8000-00805F9B0131"

    # Connect to the micro:bit
    device = await BleakScanner.find_all(timeout=5)
    if len(device) > 0:
        print(f"Found {len(device)} devices")
        for dev in device:
            print(dev)
            if dev.name == "BBC micro:bit":
                print("Found micro:bit")
                MAC_ADDRESS = dev.address
                break
        if MAC_ADDRESS == "your_microbit_mac_address":
            print("Please enter a valid MAC address")
            return
    else:
        print("No devices found")
        return
    
    async with BleakClient(MAC_ADDRESS) as client:
        try:
            # Read the data from the RX characteristic
            while True:
                data = await client.read_gatt_char(RX_CHARACTERISTIC_UUID)
                print(data.decode())
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
    
if __name__ == "__main__":
    asyncio.run(main())
