'''
Created on 2025/04/29

@author: sin
'''
from datetime import datetime
import json
import os

def datetime_update(ddict) :
    dt = datetime.now()
    ddict['second'] = dt.second
    ddict['timestamp'] = dt.timestamp()
    return int(dt.timestamp())

# Write the dictionary to the JSON file
def write_json(fname, ddict) :
    with open(fname, "w") as json_file:
        json.dump(ddict, json_file, indent=4)
        json_file.write('\n')

# Specify the file name
file_name = os.path.expanduser("~/Downloads/data.json")

if __name__ == '__main__':
    data_dict = dict()
    last = datetime_update(data_dict)
    print('started.')
    while True:
        now = datetime.now()
        if int(now.timestamp()) > last + 5 and int(now.timestamp()) % 30 == 0:
            last = datetime_update(data_dict)
            write_json(file_name, data_dict)
            print('written')