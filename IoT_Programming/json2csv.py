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
    infilename = sys.argv[1]
    outfilename = ''.join(infilename.split('.')[:-1]) + '.csv'
    #print(outfilename)
    keys = {"TIMESTAMP": "Timestamp",
            #"BME280.TC", "BME280.RH", "BME280.BP",
            "SHT40.TC": "Temp.", "SHT40.RH": "R. Humidity",
            #"BMP380.TC",
            "BMP380.BP": "B.Pressure", "VEML6030.LX": "Lux"}
    writeheader = True
    with open(infilename, mode='r') as jfile:
        for line in jfile:
            jsdict = json.loads(line)
            outstr = ''
            if writeheader :
                firstitem = True
                for k in keys:
                    if firstitem :
                        firstitem = False
                    else:
                        outstr += ','
                    outstr += keys[k]
                outstr += '\n'
                writeheader = False
            firstitem = True
            for k in keys:
                if firstitem :
                    firstitem = False
                else:
                    outstr += ','
                outstr += str(jsdict[k])
            with open(outfilename, mode='a') as csvfile :
                csvfile.write(outstr + '\n')
    print('bye.')
