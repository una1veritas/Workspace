'''
Created on 2021/09/20

@author: Sin Shimozono
'''

if __name__ == '__main__':
    import sys
    import json
    if not (len(sys.argv) > 1) :
        print('file name required.')
        exit()
    keys = [('SHT40', 'TC'), ('SHT40', 'RH'), ('BMP380', 'BP'), ('VEML6030', 'LX'), ] #[('DATETIME', 'DATE'), ('DATETIME', 'TIME'), 
            
    with open(sys.argv[1], mode='r') as jfile:
        for line in jfile:
            jsdict = json.loads(line)
            for keypair in keys:
                s = jsdict.get(keypair[0]).get(keypair[1])
                print(keypair[0], keypair[1], s)
    print('bye.')