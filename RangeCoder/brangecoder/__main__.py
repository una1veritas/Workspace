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
    
def main(infile = None):
    print(infile)
    hist = list()
    hist.append(1)
    hist_total = 1
    for i in range(255):
        hist.append(0)
    ch_count = 1
    with open(infile, "rb") as fp:
        for a_byte in bytestream(fp, 128):
            hist[a_byte] += 1
            hist_total += 1
            ch_count += 1
#             if chr(a_byte).isprintable() :
#                 print(chr(a_byte), end='')
#             else:
#                 print(hex(a_byte))
            if ch_count >= (1<<12) :
                break
    ranges = list()
    subsum = 0
    for i in range(0,len(hist)):
        ranges.append( (subsum, subsum + hist[i]) )
        subsum += hist[i]
    print(subsum)
    for ch in range(256):
        (l, r) = ranges[ch]
        if l != r :
            print( chr(ch) if chr(ch).isprintable() else hex(ch) ,l,r)
    print(2**math.ceil(math.log2(hist_total)))
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
        print(hex(left), hex(right), math.log2(left^right))
    print(left/(4096**n), right/(4096**n))
    
if __name__ == "__main__" :
    main(sys.argv[1])