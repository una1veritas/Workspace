'''
Created on 2020/10/28

@author: Sin Shimozono
'''

import sys, math

def bittream(fp, buffer_size = 32):    
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            for bitpos in range(0,8):
                yield 1 if (ch & 0x80) else 0
                ch <<= 1
        buff = fp.read(buffer_size)
    
def nibblestream(fp, buffer_size = 32):    
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            for i in range(2):
                yield (0xf0 & (ch<<(4*i)))>>4
        buff = fp.read(buffer_size)
    
def bytestream(fp, buffer_size = 32):    
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            yield ch
        buff = fp.read(buffer_size)
    
def main(infilename=None):
    print(infilename)
    hist = dict()
    for i in range(256):
        hist[i] = 1
    hist_total = 256
    with open(infilename, "rb") as fp:
        counter = 0
        for a_byte in bytestream(fp, 32):
            hist[a_byte] += 1
            hist_total += 1
            if chr(a_byte).isprintable() :
                print(chr(a_byte), end='')
            else:
                print(hex(a_byte))
            counter += 1
            if (counter & 0x07) == 0 :
                print() 
    print(hist_total, hist)
    print(2**math.ceil(math.log2(hist_total)))
    
if __name__ == "__main__" :
    main(sys.argv[1])