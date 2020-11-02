'''
Created on 2020/10/28

@author: Sin Shimozono
'''

import sys, math, os

def bitstream(fp, buffer_size = 32):    
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
   
def update_histgram(hist, total, block, max_block_size):
    for c in block:
        hist[c] += 1
        total += 1
    if len(block) == max_block_size :
        for i in range(len(hist)) :
            hist[i] = (hist[i]+1)>>1
        total >>= 1
    return (hist, total)

def main(infile = None):
    byte_bitsize = 8
    print(infile)
    hist = list()
    hist_total = 0
    # the initial histogram
    for i in range(1<<byte_bitsize):
        hist.append(1)
        hist_total += 1
    
    try:
        block_size_bits = 8
        block_size_bits_max = 12
        
        with open(infile, "rb") as fp:
            while True :
                buff = fp.read(1<<block_size_bits)
                # print('block_size = {}'.format(block_size) )
                # if the block is full (not the final block)
                if len(buff) == (1<<block_size_bits) :
                    (hist, hist_total) = update_histgram(hist, hist_total, buff, 1<<block_size_bits_max)
                    print('update '+str(hist_total))
                # setup the code_range
                subsum = 0
                code_range = [0]
                for i in range(len(hist)):
                    subsum += hist[i]
                    code_range.append(subsum)

                # encode
                left = 0
                right = 1
                bits = 0
                encodedstr = ''
                for i in range(len(buff)) :
                    ch = buff[i]
                    l, r = code_range[ch], code_range[ch+1]
                    width = right - left
                    right = (left<<block_size_bits) + (width * r)
                    left = (left<<block_size_bits) + (width * l)
                    bits += block_size_bits
#                    print( '[{}, {}), '.format(left/(1<<bits), right/(1<<bits)))
                    hbits = samehighbits(left, right, bits)
                    if len(hbits) :
                        mask = (1<<bits) - 1
                        mask >>= len(hbits)
#                        print(mask)
                        left &= mask
                        right &= mask
                        bits -= len(hbits)
                        encodedstr += hbits
#                         print('{}:[{:b}, {:b}) ({})'.format(hbits, left, right, bits))
#                         print(encodedstr)
# #                     if (left & 0xff == 0) and (right & 0xff == 0) :
# #                         left >>= 8
# #                         right >>= 8
# #                         bits -= 8
                encodedstr += '1'
                print()
                for i in range(len(encodedstr)):
                    binstr = encodedstr[i*8:(i+1)*8]
                    print(binstr + ' ', end='')
                print()
                print(len(encodedstr), 1<<block_size_bits)
                
                # termination condition
                if len(buff) != (1<<block_size_bits) :
                    print(hist_total, 1<<block_size_bits)
                    break
                else:
                    if block_size_bits < block_size_bits_max :
                        block_size_bits += 1
    except IOError:
        print('file "' + infile + '" open failed.')
        exit()
    print('hist_total = {}'.format(hist_total))
    for i in range(1<<byte_bitsize):
        if hist[i] == 1 :
            continue
        print( (chr(i) if chr(i).isprintable() else hex(i) ), hist[i])
    exit()
        
    ranges = list()
    subsum = 0
    for i in range(0,len(hist)):
        ranges.append( (subsum, subsum + hist[i]) )
        subsum += hist[i]
#     for ch in range(256):
#         (l, r) = ranges[ch]
#          if l != r :
#              print( chr(ch) if chr(ch).isprintable() else hex(ch) , r - l)
    print(hist_total, 2**math.ceil(math.log2(hist_total)))
    
    with open(infile, "rb") as fp:
        code = list()
        for a_byte in bytestream(fp, 256):
            #print(ranges[a_byte])
            code.append(ranges[a_byte])
    code.append(ranges[0])
    print('file size = '+str(len(code)))
    left = 0
    right = 1
    bits = 0
    encodedstr = ''
    for (l, r) in code[:4096] :
#        print('---–\n{0}\n{1}\n'.format(format(left, 'b').zfill(bits), format(right, 'b').zfill(bits)))
        hbits = samehighbits(left, right, bits)
        if len(hbits) :
            mask = (1<<bits) - 1
            mask >>= len(hbits)
            left &= mask
            right &= mask
            bits -= len(hbits)
#             print('{}.{}\n{}.{}\n({}:{})\n'.format(hbits, format(left,'b').zfill(bits), hbits, format(right, 'b').zfill(bits), len(hbits), bits))
            encodedstr += hbits
        if (left & 0xff == 0) and (right & 0xff == 0) :
            left >>= 8
            right >>= 8
            bits -= 8
    else:
        encodedstr += '1'

    #print('{}\n{}'.format(bin(right), len(bin(right))))
    print(encodedstr)
    print(len(encodedstr))
    print('finished.')
    
if __name__ == "__main__" :
    main(sys.argv[1])