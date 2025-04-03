'''
Created on 2025/03/15

@author: sin
'''
# SPDX-FileCopyrightText: 2020 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT

"""
This example scans for any BLE advertisements and prints one advertisement and one scan response
from every device found.
"""

from adafruit_ble import BLERadio

from adafruit_ble.advertising import Advertisement
from adafruit_ble.advertising.standard import ProvideServicesAdvertisement


print("scan done")
if __name__ == '__main__':
    ble = BLERadio()
    print("scanning")
    found = set()
    scan_responses = set()
    for advertisement in ble.start_scan(ProvideServicesAdvertisement, Advertisement):
        addr = advertisement.address
        if advertisement.scan_response and addr not in scan_responses:
            scan_responses.add(addr)
        elif not advertisement.scan_response and addr not in found:
            found.add(addr)
        else:
            continue
        print(addr.string, advertisement.complete_name)
        #print("\t" + repr(advertisement))
        print()
    print('scan done.')
    print(f'found = {found}')
    print(f'scan_responses = {scan_responses}')