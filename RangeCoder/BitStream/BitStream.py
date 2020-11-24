'''
Created on 2020/11/24

@author: sin
'''

class BitStream(object):
    '''
    int bval
    int blength
    '''


    def __init__(self, val = 0, length = 0):
        '''
        Constructor
        '''
        self.bval = val
        if length > 0 :
            self.blength = length
        elif self.bval > 0 :
            digits = 0
            tval = self.bval
            while tval > 0 :
                digits += 1
                tval >>= 1
            self.blength = digits
        else:
            self.blength = 0
    
    def __str__(self):
        if not self.blength > 0 :
            return ''
        bmask = 1<<(self.blength-1)
        tmpstr = ''
        while bmask > 0 :
            tmpstr += '1' if bmask & self.bval else '0'
            bmask >>= 1
        return tmpstr
    
    def __int__(self):
        return self.bval

    def append(self, bits, length = 0):
        if type(bits) is int :
            self.bval <<= length
            self.bval |= bits
            self.blength += length
            return self
        elif type(bits) is str:
            for d in bits:
                if d == '0' :
                    self.bval <<= 1
                    self.blength += 1
                else:
                    self.bval <<= 1
                    self.bval |= 1
                    self.blength += 1
            return self

    def retract(self, bits = 8):
        if bits >= self.blength :
            t = self.bval
            self.blength = 0
            self.bval = 0
            return t
        t = 0
        mask = 1<<(self.blength-1)
        while  bits > 0:
            t <<= 1
            if (self.bval & mask) != 0 :
                t |= 1
            self.blength -= 1
            self.bval &= (~mask)
            bits -= 1
            mask >>= 1
        return t
                      
bs = BitStream(0, 9)
print(bs)
print(bs.append('10110'))
print(bs.append(7, 5))
print(int(bs))
print(bin(bs.retract(9)), bs)
print(bin(bs.retract(9)), bs)
print(bin(bs.retract(9)), bs)
