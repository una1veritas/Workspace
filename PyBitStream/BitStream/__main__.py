'''
Created on 2018/06/24

@author: Sin Shimozono
'''

import sys

class BitStream():
    defaultBufferSize = 512
    def __init__(self):
        self.buffsize = self.defaultBufferSize
        self.buffer = list()
    
    def length(self):
        return len(self.buffer)
    
    def bitIn(self, val):
        self.buffer.append(1 if (val & 1) else 0)
        
    def byteIn(self, val):
        for bpos in [128, 64, 32, 16, 8, 4, 2, 1]:
            self.buffer.append(1 if val & bpos else 0)
    
    def alphaIn(self,val,terminator=1):
        if val < 0:
            return
        for __ in range(1,val):
            self.buffer.append(terminator-1)
        self.buffer.append(terminator)
        
    def gammaIn(self, val):
        blen = max(val.bit_length(),1)
        self.alphaIn(blen,1)
        bdigit = 1 << (blen-1)
        bdigit = max(bdigit>>2, 1)
        while bdigit > 0:
            if val & bdigit :
                self.buffer.append(1)
            else:
                self.buffer.append(0)
            bdigit = bdigit>>1

    def deltaIn(self, val):
        blen = max(val.bit_length(),1)
        self.gammaIn(blen)
        bdigit = 1 << (blen-1)
        bdigit = max(bdigit>>2, 1)
        while bdigit > 0:
            if val & bdigit :
                self.buffer.append(1)
            else:
                self.buffer.append(0)
            bdigit = bdigit>>1

    def bitOut(self):
        bval = self.buffer.pop(0)
        return bval

    def bitsOut(self, digits = 1):
        result = [ ]
        for __ in range(0, digits):
            result.append( self.buffer.pop(0) )
        return result
    
CHUNKSIZE = 64

if __name__ == '__main__':
    pass

bstream = BitStream()

for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31,32]:
    print(d, end= ' ')
    bstream.deltaIn(d)
    while bstream.length() :
        print(bstream.bitOut(), end='')
    print()


fname = sys.argv[1]
bfile = open(fname, "rb")
last_byte = 0
try:
    while (True) :
        bytes_read = bfile.read(CHUNKSIZE)
        if not bytes_read:
            break
        for b in bytes_read:
            bstream.deltaIn(b)
            print(hex(b), end= ' ')
        print()
        while (bstream.length() > 0):
            print(bstream.bitOut(), end='')
        print()
finally:
    bfile.close()
    
    