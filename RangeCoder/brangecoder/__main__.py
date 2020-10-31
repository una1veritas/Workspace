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

def samehighbits(l, r, w): 
    x = l^r
    for b in range(w):
        if (x>>(w-1-b)) & 1 :
            break
#    print('w='+str(w)+', b='+str(b))
    hbits = ''
    if not b :
        return hbits
    for i in range(w):
        if i < b :
            if l>>(w-1-i) & 1 :
                hbits += '1'
            else:
                hbits += '0'
        else:
            break
    return hbits
   
def main(infile = None):
    char_bitsize = 8
    chunk_bitsize = 12
    print(infile)
    hist = list()
    for i in range(1<<char_bitsize):
        hist.append(1)
    hist_total = 1<<char_bitsize
    with open(infile, "rb") as fp:
        for a_byte in bytestream(fp, 256):
            hist[a_byte] += 1
            hist_total += 1
#             if chr(a_byte).isprintable() :
#                 print(chr(a_byte), end='')
#             else:
#                 print(hex(a_byte))
            if hist_total >= (1<<chunk_bitsize) :
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
    bits = 0
    encodedstr = ''
    for i in range(32):
        (l, r) = code[i]
        width = right - left
        right = (left<<chunk_bitsize) + (width * r)
        left = (left<<chunk_bitsize) + (width * l)
        bits += chunk_bitsize
#        print('---–\n{0}\n{1}\n'.format(format(left, 'b').zfill(bits), format(right, 'b').zfill(bits)))
        hbits = samehighbits(left, right, bits)
        if len(hbits) :
            mask = (1<<bits) - 1
            mask >>= len(hbits)
            left &= mask
            right &= mask
            bits -= len(hbits)
            print('{}.{}\n{}.{} ({}:{})\n'.format(hbits, format(left,'b').zfill(bits), hbits, format(right, 'b').zfill(bits), len(hbits), bits))
            encodedstr += hbits

    print(left, right)
    print(encodedstr, len(encodedstr))
    
if __name__ == "__main__" :
    main(sys.argv[1])