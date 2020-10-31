'''
Created on 2020/10/28

@author: Sin Shimozono
'''

import sys, math

def bittream(fp, buffer_size = 32):    
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            for bp in range(0,8):
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
    
def bytestream(fp, buffer_size = 32) :
    buff = fp.read(buffer_size)
    while buff :
        for ch in buff:
            yield ch
        buff = fp.read(buffer_size)

def samehighbits(l, r): 
    x = l^r
    d = math.floor(math.log2(l | r))+1
    for i in range(1,d):
        if (x>>(d-i)) & 1 == 1 :
            break
    return i-1
    
def main(infile = None):
    print(infile)
    hist = list()
    for i in range(256):
        hist.append(1)
    hist_total = 256
    cnt = 256
    with open(infile, "rb") as fp:
        for a_byte in bytestream(fp, 256):
            hist[a_byte] += 1
            hist_total += 1
#             if chr(a_byte).isprintable() :
#                 print(chr(a_byte), end='')
#             else:
#                 print(hex(a_byte))
            cnt += 1
            if cnt >= 4096 :
                break
    ranges = list()
    subsum = 0
    for i in range(0,len(hist)):
        ranges.append( (subsum, subsum + hist[i]) )
        subsum += hist[i]
    for ch in range(256):
        (l, r) = ranges[ch]
#         if l != r :
#             print( chr(ch) if chr(ch).isprintable() else hex(ch) , r - l)
    print(hist_total, 2**math.ceil(math.log2(hist_total)))
    
    with open(infile, "rb") as fp:
        code = list()
        for a_byte in bytestream(fp, 128):
            #print(ranges[a_byte])
            code.append(ranges[a_byte])
    code.append(ranges[0])
    print(len(code))
    left = 0
    right = 1
    n = 12
    for i in range(8):
        (l, r) = code[i]
        width = right - left
        right = left*4096 + (width * r)
        left = left*4096 + (width * l)
        print('---–\n{0:b}\n{1:b}\n{2:}\n'.format(left, right, samehighbits(left, right)))
    print('\n',left, right)
    
if __name__ == "__main__" :
    main(sys.argv[1])